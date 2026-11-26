#!/usr/bin/env bash
# Prepare this Spark to serve Inkling-2-Mini (default) or, with RERISKED=1, Inkling-2-Mini-Derisked with TensorFold's Zig engine (tensorfold-native, one GPU):
#   1. preflight: docker, the GPU, memory, disk
#   2. the image: tensorfold-native built from TensorFold (TF_REPO at TF_REF, PR #4441's Inkling 2 Mini CUDA engine) plus
#      patches/*.patch with Zig, on NVIDIA's PyTorch container (kernels are embedded in the binary), with libtfgrammar.so
#      built from the same patched tree over xgrammar v0.2.8 (commit pinned), and the JPEG and WebP decoders for
#      image inputs (libjpeg-turbo 3.1.4.1, libwebp 1.6.0).
#   3. the checkpoint in MODEL_DIR: upstream thinkingmachines/Inkling-2-Mini (pinned revision) is fetched by
#      scripts/fetch.sh when incomplete; RERISKED=1's local Derisked folder is never downloaded. Its config must be
#      inkling2_loop, and the target shard and mtp.safetensors must match their pinned SHA-256. The hashes are taken
#      once and kept in STATE_DIR by size and mtime, so later runs skip them.
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch, --rehash to hash the weights again.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REBUILD=0
REHASH=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    --rehash) REHASH=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it"

log "Preflight checks"
_mem_here=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
if (( _mem_here < MEM_FLOOR_GIB )); then
  _msg="only ${_mem_here} GiB MemAvailable (the floor is MEM_FLOOR_GIB=$MEM_FLOOR_GIB): stop other work first"
  if [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$_msg"; else die "$_msg (MEM_CHECK=0 continues anyway)"; fi
fi
log "Memory: ${_mem_here} GiB MemAvailable (floor $MEM_FLOOR_GIB)"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime; --gpus all may fail"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
have=$(free_gb "$DOCKER_ROOT"); have=${have:-0}
(( have >= need_img )) || die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB"
log "Disk: ${have} GB free under Docker's root"

if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }
  log "Building $IMAGE (TensorFold ${TF_REF:0:12} from $TF_REPO, Zig $ZIG_VERSION, patches $PATCHES_HASH)"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches"
  compgen -G 'patches/*.patch' >/dev/null && cp -- patches/*.patch "$ctx/patches/"
  if [[ "$RUNNER" == 1 ]]; then python3 -I scripts/runner_assets.py "$ctx/runner" || die "the runner assets did not check out"
  else mkdir "$ctx/runner"; fi
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  # shellcheck disable=SC2086  # DOCKER_BUILD_ARGS is a list of flags on purpose
  docker build ${DOCKER_BUILD_ARGS:-} "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg XGRAMMAR_REPO="$XGRAMMAR_REPO" --build-arg XGRAMMAR_TAG="$XGRAMMAR_TAG" --build-arg XGRAMMAR_REF="$XGRAMMAR_REF" \
    --build-arg NATIVE_FP_DECODE="$NATIVE_FP_DECODE" --build-arg PATCHES_HASH="$PATCHES_HASH" --build-arg RUNNER="$RUNNER" \
    -f - "$ctx" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
# PNG is built into the engine; JPEG and WebP images go through these two pinned decoders, which the engine loads from
# <binary dir>/../lib. aarch64: libjpeg-turbo's NEON code builds with gcc's assembler, so no nasm.
FROM ${BASE_IMAGE} AS codecs
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake make && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/codecs/lib /opt/codecs/doc && \
    curl -fsSL --retry 3 -o /tmp/libjpeg-turbo.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz" && \
    tar -xzf /tmp/libjpeg-turbo.tar.gz -C /tmp && \
    cmake -S /tmp/libjpeg-turbo-3.1.4.1 -B /tmp/build-jpeg -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON && \
    cmake --build /tmp/build-jpeg --target turbojpeg -j"$(nproc)" && \
    cp -a /tmp/build-jpeg/libturbojpeg.so.0* /opt/codecs/lib/ && \
    cp /tmp/libjpeg-turbo-3.1.4.1/LICENSE.md /opt/codecs/doc/libjpeg-turbo-LICENSE.md && \
    curl -fsSL --retry 3 -o /tmp/libwebp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz" && \
    tar -xzf /tmp/libwebp.tar.gz -C /tmp && \
    cmake -S /tmp/libwebp-1.6.0 -B /tmp/build-webp -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_LIBWEBPMUX=OFF && \
    cmake --build /tmp/build-webp --target webpdecoder -j"$(nproc)" && \
    cp -a /tmp/build-webp/libwebpdecoder.so.3* /opt/codecs/lib/ && \
    cp /tmp/libwebp-1.6.0/COPYING /opt/codecs/doc/libwebp-BSD-3-Clause.txt && \
    rm -rf /tmp/libjpeg-turbo* /tmp/build-jpeg /tmp/libwebp* /tmp/build-webp

FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# fatbins + native only: the kernels are embedded in tensorfold-native (images: the codecs stage; no video)
ARG NATIVE_FP_DECODE=1
RUN cd /tensorfold && NFP=false && if [ "${NATIVE_FP_DECODE}" = 1 ]; then NFP=true; fi && \
    /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast -Dnative-fp-decode=$NFP --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
# RUNNER=1: the circuit frontend's boot self-test (compiles a C cell to an RV32 ELF; the engine runs the same check)
ARG RUNNER=0
RUN if [ "${RUNNER}" = 1 ]; then cd /tensorfold && /opt/zig/zig build runner-frontend-check --prefix /opt/tensorfold -j"$(nproc)" && \
      test -x /opt/tensorfold/bin/tf-runner-frontend-check; fi
# libtfgrammar.so (structured output, required tool calls) from the patched tree's tf_grammar.cc over xgrammar
# v0.2.8 at its pinned commit; the engine finds it in <binary dir>/../lib, i.e. /opt/tensorfold/native/lib
ARG XGRAMMAR_REPO
ARG XGRAMMAR_TAG
ARG XGRAMMAR_REF
RUN git clone -q --depth 1 --branch "${XGRAMMAR_TAG}" --recurse-submodules --shallow-submodules "${XGRAMMAR_REPO}" /opt/xgrammar && \
    test "$(git -C /opt/xgrammar rev-parse HEAD)" = "${XGRAMMAR_REF}" && \
    cd /tensorfold && /opt/zig/zig build grammar -Dxgrammar=/opt/xgrammar -Doptimize=ReleaseFast --prefix /opt/tensorfold -j"$(nproc)" && \
    test -s /opt/tensorfold/native/lib/libtfgrammar.so && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar && \
    cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar/xgrammar-Apache-2.0.txt && \
    cp /opt/xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar/xgrammar-NOTICE.txt && \
    cp /opt/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar/dlpack-Apache-2.0.txt
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done

FROM ${BASE_IMAGE}
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/lib/ /opt/tensorfold/native/lib/
COPY --from=codecs /opt/codecs/doc/ /opt/tensorfold/share/doc/tensorfold/LICENSES-codecs/
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
# RUNNER=1: clang-18 and lld-18 for the runner's RV32 programs, its circuit and graph bank, and a check that the
# circuit computes the engine's probe cells; RUNNER=0 leaves the image as without it
ARG RUNNER=0
COPY runner /opt/tensorfold/share/runner
RUN if [ "${RUNNER}" = 1 ]; then \
      apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-18 lld-18 && \
      ln -sf clang-18 /usr/bin/clang && ln -sf ld.lld-18 /usr/bin/ld.lld && rm -rf /var/lib/apt/lists/* && \
      mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES-runner && \
      cp /opt/tensorfold/share/runner/graph-v7-powmod/bank/Qwenputer-MIT.txt /opt/tensorfold/share/doc/tensorfold/LICENSES-runner/ && \
      /opt/tensorfold/bin/tf-runner-frontend-check --require --circuit /opt/tensorfold/share/runner/calc.tfpc; \
    else rmdir /opt/tensorfold/share/runner; fi
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.kernels=embedded
ENV CUDA_CACHE_PATH=/cache/nv
WORKDIR /workspace
DOCKERFILE
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built with patches $PATCHES_HASH"
fi
docker run --rm --network none --entrypoint test "$IMAGE" -x /opt/tensorfold/native/bin/tensorfold-native ||
  die "$IMAGE has no tensorfold-native"
log "Image $IMAGE: tensorfold-native, kernel set present"

if [[ "${MODEL_FETCH:-0}" == 1 ]] && ! [[ -s "$MODEL_DIR/$SHARD_NAME" && -s "$MODEL_DIR/$MTP_NAME" && -s "$MODEL_DIR/config.json" ]]; then
  log "$MODEL_LABEL is not complete in $MODEL_DIR: fetching it (about 20 GB, resumable)"
  ./scripts/fetch.sh "$MODEL_DIR"
fi
if [[ "$DFLASH2" == 1 && "${DRAFTER_FETCH:-0}" == 1 ]] && ! [[ -s "$DRAFTER_DIR/$DRAFTER_NAME" && -s "$DRAFTER_DIR/config.json" ]]; then
  log "The DFlash2 draft model is not complete in $DRAFTER_DIR: fetching it (about 1 GB, resumable)"
  ./scripts/fetch.sh drafter "$DRAFTER_DIR"
fi
log "Verifying the checkpoint at $MODEL_DIR ($MODEL_LABEL)"
[[ -d "$MODEL_DIR" ]] || die "MODEL_DIR=$MODEL_DIR is not a directory: point it at the $MODEL_LABEL folder"
CHECK_PY='
import json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "inkling2_loop":
    sys.exit("model_type is %r, not inkling2_loop" % cfg.get("model_type"))
q = cfg.get("quantization_config") or {}
if str(q.get("quant_method", "")).lower() != "modelopt":
    sys.exit("not a ModelOpt checkpoint (quant_method %r)" % q.get("quant_method"))
t = cfg.get("text_config") or {}
loop = t.get("loop_config") or {}
if not cfg.get("mtp_config"):
    sys.exit("config.json has no mtp_config: the DSpark drafter needs it")
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
if bad:
    sys.exit("missing or empty shards: " + ", ".join(bad[:5]))
print("inkling2_loop, ModelOpt %s, %d layers, layers %s-%s looped %s times, %d tensors in %d shards" % (
    q.get("quant_algo"), t.get("num_hidden_layers", 0), loop.get("core_start"), loop.get("core_end", 0) - 1,
    loop.get("num_passes"), len(idx["weight_map"]), len(shards)))'
info=$(python3 -I -c "$CHECK_PY" "$MODEL_DIR" 2>&1) || die "the checkpoint at $MODEL_DIR is not ready: $info"
log "Checkpoint OK: $info"
for f in "$SHARD_NAME" "$MTP_NAME"; do
  [[ -s "$MODEL_DIR/$f" ]] || die "$MODEL_DIR/$f is missing or empty"
done
if [[ "$DFLASH2" == 1 ]]; then
  [[ -s "$DRAFTER_DIR/$DRAFTER_NAME" && -s "$DRAFTER_DIR/config.json" ]] ||
    die "$DRAFTER_DIR holds no DFlash2 draft model (config.json, $DRAFTER_NAME): scripts/fetch.sh drafter, set DRAFTER_DIR, or DRAFTER=dspark"
  python3 -I -c 'import json,sys; c=json.load(open(sys.argv[1])); d=c.get("dflash_config") or sys.exit("no dflash_config");
sys.exit(0 if d.get("target_layer_ids")==[5,23,41,48] and d.get("block_size")==8 else "not the Inkling-2-Mini DFlash2 draft")' "$DRAFTER_DIR/config.json" ||
    die "$DRAFTER_DIR/config.json is not incoai/Inkling-2-Mini-DFlash2's"
  log "DFlash2 draft OK: $DRAFTER_DIR ($DRAFTER_REPO@${DRAFTER_REV:0:12})"
fi
stamp=$(weights_stamp)
if [[ $REHASH -eq 0 && "$(cat "$VERIFIED_MARKER" 2>/dev/null)" == "$stamp" ]]; then
  log "SHA-256 already verified for these files (same size and mtime; --rehash checks again)"
else
  log "Hashing $SHARD_NAME (~18 GiB) and $MTP_NAME; this takes a minute"
  got=$(sha256sum "$MODEL_DIR/$SHARD_NAME" | cut -d' ' -f1)
  [[ "$got" == "$SHARD_SHA256" ]] || die "$SHARD_NAME SHA-256 is $got, not $SHARD_SHA256 (pinned for $MODEL_LABEL): not this checkpoint"
  got=$(sha256sum "$MODEL_DIR/$MTP_NAME" | cut -d' ' -f1)
  [[ "$got" == "$MTP_SHA256" ]] || die "$MTP_NAME SHA-256 is $got, not $MTP_SHA256 (the upstream MTP module's)"
  if [[ "$DFLASH2" == 1 ]]; then
    got=$(sha256sum "$DRAFTER_DIR/$DRAFTER_NAME" | cut -d' ' -f1)
    [[ "$got" == "$DRAFTER_SHA256" ]] || die "$DRAFTER_DIR/$DRAFTER_NAME SHA-256 is $got, not $DRAFTER_SHA256 ($DRAFTER_REPO@${DRAFTER_REV:0:12})"
  fi
  printf '%s' "$stamp" > "$VERIFIED_MARKER"
  log "SHA-256 OK: target shard ${SHARD_SHA256:0:12}, MTP ${MTP_SHA256:0:12}$( [[ "$DFLASH2" == 1 ]] && echo ", DFlash2 ${DRAFTER_SHA256:0:12}")"
fi
prepared_state > "$PREPARED_MARKER"
log "Done. Start the server with ./start.sh (port $PORT)."
