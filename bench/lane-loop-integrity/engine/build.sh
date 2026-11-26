#!/usr/bin/env bash
# The lane-loop-integrity test image on a Spark, from the recipe checkout: the shipped build tree (scripts/prepare.sh's
# build stage) in a long-lived container, then this folder's 0001-state-sums.patch rebuilt over it, and the shipped
# image with only tensorfold-native replaced.
#   bash bench/lane-loop-integrity/engine/build.sh [base|hook|image|all]   (default all)
# LLI_WORK: host work dir (default $HOME/work/lli-hook). LLI_BUILD: the build container (lli-hook-build).
# LLI_BASE: the shipped image (tensorfold-inkling2:c4-<ref>-runner). LLI_IMAGE: the test image (LLI_BASE-lli).
# LLI_GIT_PROXY: a proxy for the engine fetch (a Spark without a route to TF_REPO: `ssh -R 11080 <spark>` and
# LLI_GIT_PROXY=socks5h://127.0.0.1:11080).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# shellcheck source=SCRIPTDIR/../../../scripts/config.sh
source scripts/config.sh

here=bench/lane-loop-integrity/engine
step="${1:-all}"
[[ "$step" =~ ^(base|hook|image|all)$ ]] || die "step is base, hook, image or all, not $step"
work="${LLI_WORK:-$HOME/work/lli-hook}"
build="${LLI_BUILD:-lli-hook-build}"
shipped="${LLI_BASE:-tensorfold-inkling2:c4-${TF_REF:0:7}-runner}"
out_image="${LLI_IMAGE:-${shipped}-lli}"
nfp=false; [[ "$NATIVE_FP_DECODE" == 1 ]] && nfp=true
bin=/opt/tensorfold/native/bin/tensorfold-native

mkdir -p "$work/in/patches" "$work/out"
find "$work/in/patches" -maxdepth 1 -name '*.patch' -delete
cp -- patches/*.patch "$work/in/patches/"
if [[ "$step" != base ]]; then cp -- "$here/0001-state-sums.patch" "$work/in/state-sums.patch"; fi

if ! docker container inspect "$build" >/dev/null 2>&1; then
  log "Starting $build from $BASE_IMAGE"
  docker run -d --name "$build" --network host -v "$work/in:/in:ro" "$BASE_IMAGE" sleep infinity >/dev/null
fi
docker start "$build" >/dev/null

in_build() { docker exec -i -e ZIG_VERSION="$ZIG_VERSION" -e ZIG_SHA256="$ZIG_SHA256" -e TF_REPO="$TF_REPO" -e TF_REF="$TF_REF" -e NFP="$nfp" -e GIT_PROXY="${LLI_GIT_PROXY:-}" "$build" env -u BASH_ENV bash -euo pipefail -c "$1"; }

# the build stage exactly: Zig at its pin, the engine at TF_REF, patches/*.patch in order, fatbins and native
base_build() {
  # shellcheck disable=SC2016  # expanded in the container
  in_build '
    if [ ! -x /opt/zig/zig ]; then
      curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"
      echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -
      mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
    fi
    /opt/zig/zig version
    if [ ! -e /tensorfold/.lli-patched ]; then
      rm -rf /tensorfold
      git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}"
      git ${GIT_PROXY:+-c http.proxy="$GIT_PROXY"} fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD
      test "$(git rev-parse HEAD)" = "${TF_REF}"
      for p in /in/patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
      git add -A && git -c user.name=lli -c user.email=lli@localhost commit -qm patched && git tag -f patched >/dev/null
      touch /tensorfold/.lli-patched
    fi
    cd /tensorfold
    if [ ! -e /opt/lli-shipped-build ]; then
      /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast -Dnative-fp-decode=$NFP --prefix /opt/tensorfold \
        -j"$(nproc)" fatbins native
      test -x /opt/tensorfold/native/bin/tensorfold-native
      sha256sum /opt/tensorfold/native/bin/tensorfold-native | tee /opt/lli-shipped-build
    fi
    cat /opt/lli-shipped-build'
}

# the hook over the patched tree (the tree reset to the tag first), rebuilt incrementally
hook_build() {
  # shellcheck disable=SC2016  # expanded in the container
  in_build '
    cd /tensorfold
    git reset -q --hard patched && rm -f zig/src/families/inkling2/mini_sums.zig
    git apply --check /in/state-sums.patch && git apply /in/state-sums.patch
    /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast -Dnative-fp-decode=$NFP --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native
    sha256sum /opt/tensorfold/native/bin/tensorfold-native'
}

make_image() {
  docker image inspect "$shipped" >/dev/null 2>&1 || die "no $shipped here: build it with scripts/prepare.sh (RUNNER=1)"
  docker cp "$build:$bin" "$work/out/tensorfold-native"
  printf 'FROM %s\nCOPY tensorfold-native %s\n' "$shipped" "$bin" >"$work/out/Dockerfile"
  docker build -q -t "$out_image" "$work/out" >/dev/null
  log "$out_image $(docker image inspect -f '{{.Id}}' "$out_image")"
  log "tensorfold-native $(sha256sum "$work/out/tensorfold-native" | cut -d' ' -f1)"
}

case "$step" in
  base) base_build ;;
  hook) hook_build ;;
  image) make_image ;;
  all) base_build; hook_build; make_image ;;
esac
