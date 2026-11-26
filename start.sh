#!/usr/bin/env bash
# Serve Inkling-2-Mini (default; RERISKED=1 serves the local Inkling-2-Mini-Derisked folder) with TensorFold's Zig engine on one DGX Spark: runs
# scripts/prepare.sh when the image or the checkpoint check is not current, launches tensorfold-native serve on port
# 8888 with the DFlash2 draft model (incoai/Inkling-2-Mini-DFlash2; DRAFTER=dspark: the checkpoint's own DSpark head),
# waits until the OpenAI API answers, then runs a smoke test.
# Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                         # 2 requests at once, a 131,072-token window, DFlash2 drafts, states on disk
#   ./start.sh restart                 # stop, then start again; the new arguments are checked before stopping
#   ./start.sh restart --parallel 2 --context 131072
#   CONTEXT=131072 ./start.sh restart  # a shorter window leaves memory for more streams or the prompt cache
#   DRAFTS=0 ./start.sh restart        # no drafter (--no-drafts)
#   DRAFTER=dspark ./start.sh restart  # the checkpoint's own DSpark head instead of DFlash2
#   VISION=0 ./start.sh restart        # text only: no vision tower (saves ~0.6 GiB)
#   COMPUTE_LANE=1 ./start.sh restart  # run the model's code_interpreter calls in a sandbox, inside the reply
#   DRY_RUN=1 ./start.sh               # print the docker command and exit
# Extra arguments go to tensorfold-native serve after the defaults, so they win.
# Settings, from the environment or ./.env (scripts/config.sh):
#   serving  CONTEXT, PARALLEL, MAX_TOKENS, THINKING, REASONING_EFFORT, DRAFTS, DRAFTER, VISION, VISION_URLS, PROMPT_CACHE_DISK,
#            COMPUTE_LANE, VISION_MAX_IMAGES, VISION_IMAGE_TOKENS, TEMPERATURE, TOP_P, TOP_K, SERVED_NAME, HOST, PORT
#   files    MODEL_DIR, DRAFTER_DIR, KERNEL_CACHE, STATE_DIR, SESSION_DIR
#   image    IMAGE, TF_REPO, TF_REF, ZIG_VERSION, BASE_IMAGE, NATIVE_FP_DECODE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), FOREGROUND=1, WAIT_TIMEOUT, DRY_RUN=1, MEM_NEED_GIB, MEM_CHECK=0
#   engine   every TENSORFOLD_*, TF_INKLING2_*, TF_NVFP4_*, TF_REASONING_*, TF_TOOL_CALL_* and TF_KPROF* variable goes into the
#            container (TENSORFOLD_API_KEY by name only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count up to 1048576 (the model's window), not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
for v in THINKING DRAFTS VISION VISION_URLS PROMPT_CACHE_DISK COMPUTE_LANE; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ -z "$REASONING_EFFORT" || "$REASONING_EFFORT" =~ ^(none|minimal|low|medium|high|xhigh|max|0(\.[0-9]+)?)$ ]] ||
  die "REASONING_EFFORT is none, minimal, low, medium, high, xhigh, max or a number in [0, 0.99], not $REASONING_EFFORT"
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
if [[ "$DRAFTS" != 1 ]]; then SERVE_ARGS+=(--no-drafts)
elif [[ "$DRAFTER" == dflash2 ]]; then SERVE_ARGS+=(--drafter "$DRAFTER_MOUNT")
else SERVE_ARGS+=(--drafter "$MODEL_MOUNT"); fi
if [[ "$VISION" == 1 ]]; then
  SERVE_ARGS+=(--vision)
  [[ "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
  [[ -z "$VISION_MAX_IMAGES" ]] || SERVE_ARGS+=(--vision-max-images "$VISION_MAX_IMAGES")
  [[ -z "$VISION_IMAGE_TOKENS" ]] || SERVE_ARGS+=(--vision-image-tokens "$VISION_IMAGE_TOKENS")
fi
[[ "$KV_DTYPE" =~ ^(bf16|fp8)$ ]] || die "KV_DTYPE is bf16 or fp8, not $KV_DTYPE"
[[ "$KV_DTYPE" == bf16 ]] || SERVE_ARGS+=(--kv-cache "$KV_DTYPE")
[[ "$PROMPT_CACHE_DISK" == 1 ]] && SERVE_ARGS+=(--prompt-cache-disk /sessions)
USER_ARGS=("$@")
arg_value() {
  local flag=$1 value="" i
  local -a all=("${SERVE_ARGS[@]}" "${USER_ARGS[@]}")
  for (( i = 0; i < ${#all[@]}; i++ )); do
    case "${all[i]}" in
      "$flag") value="${all[i + 1]:-}" ;;
      "$flag="*) value="${all[i]#*=}" ;;
    esac
  done
  echo "$value"
}
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · %s at once · %s-token window · %s KV · drafts %s · port %s%s\n\n' "$D" "$MODEL_LABEL" \
  "$(arg_value --parallel)" "$(arg_value --context)" "$KV_DTYPE" "$(drafts_label)" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
[[ "$PROMPT_CACHE_DISK" == 1 ]] && mkdir -p "$SESSION_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
(( DRY )) && log "DRY_RUN=1: printing the docker command; nothing is stopped or started"

if (( ! DRY )) && [[ "$MODE" == start ]] && running; then
  log "$CONTAINER_NAME is already running (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it, or ./stop.sh to stop it."
  exit 0
fi

step 1 "Setup: image and checkpoint"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now"
  else
    log "Not ready yet: running scripts/prepare.sh"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_DIR"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
(( DRY )) && why="DRY_RUN: scripts/prepare.sh would"
missing() { if (( DRY )); then warn "$1"; else die "$1"; fi; }
have_image=1
docker image inspect "$IMAGE" >/dev/null 2>&1 || { have_image=0; missing "image $IMAGE missing: $why build it"; }
if (( have_image )); then
  kernels=$(docker image inspect -f '{{index .Config.Labels "tf.kernels"}}' "$IMAGE" 2>/dev/null || true)
  [[ "$kernels" == embedded ]] || missing "$IMAGE is not an image of this recipe (tf.kernels=${kernels:-?}): scripts/prepare.sh --rebuild"
fi
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(image_hash)
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/$MTP_NAME" ]] || missing "$MODEL_DIR holds no Inkling-2-Mini checkpoint (config.json, $MTP_NAME): set MODEL_DIR"
DRAFT_MOUNT=()
if [[ "$DFLASH2" == 1 ]]; then
  [[ -f "$DRAFTER_DIR/config.json" && -f "$DRAFTER_DIR/$DRAFTER_NAME" ]] ||
    missing "$DRAFTER_DIR holds no DFlash2 draft model (config.json, $DRAFTER_NAME): scripts/fetch.sh drafter, set DRAFTER_DIR, or DRAFTER=dspark"
  DRAFT_MOUNT=(-v "$DRAFTER_DIR":"$DRAFTER_MOUNT":ro)
fi

step 2 "Checks: arguments, previous server, port, memory"
if (( have_image )); then
  code=0
  docker run --rm --network none --entrypoint /opt/tensorfold/native/bin/tensorfold-native "$IMAGE" \
    serve /nonexistent/argument-check --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" "${USER_ARGS[@]}" \
    >/dev/null 2>"$STATE_DIR/args.err" || code=$?
  if (( code == 2 )) || ! grep -q 'neither a directory' "$STATE_DIR/args.err"; then
    if (( DRY )); then warn "DRY_RUN: tensorfold-native rejects these arguments: $(grep -v '^usage:' "$STATE_DIR/args.err" | tail -1)"
    else cat "$STATE_DIR/args.err" >&2; die "tensorfold-native serve rejects these arguments; nothing was changed"; fi
  fi
fi
if (( DRY )); then
  running && log "DRY_RUN: $CONTAINER_NAME is running; a real start would stop it first"
elif [[ "$MODE" == restart ]] && running; then
  ./stop.sh
elif [[ "$MODE" == start ]] && running; then
  log "A server was left running: stopping it, then starting"
  ./stop.sh
fi
if (( ! DRY )) && docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Removing the previous container $CONTAINER_NAME"
  mkdir -p "$LOG_DIR"
  docker logs "$CONTAINER_NAME" > "$LOG_DIR/$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  if (( DRY )); then log "DRY_RUN: port $PORT is in use"
  else die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"; fi
fi
need_gib=${MEM_NEED_GIB:-$(mem_need_gib "$(arg_value --context)")}
avail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)   # not free(1): its row labels are translated
if (( avail_gb >= need_gib )); then
  log "Arguments OK, port $PORT free, ${avail_gb} GiB memory available (this window needs ~$need_gib at start)"
else
  msg="only ${avail_gb} GiB memory available; $(arg_value --parallel) windows of $(arg_value --context) tokens need ~$need_gib (MemAvailable)"
  if (( DRY )) || [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$msg"; else die "$msg (MEM_CHECK=0 starts anyway)"; fi
fi

SESSION_MOUNT=(); [[ "$PROMPT_CACHE_DISK" == 1 ]] && SESSION_MOUNT=(-v "$SESSION_DIR":/sessions)
[[ "$COMPUTE_LANE" == 1 && -z "${TF_INKLING2_COMPUTE_LANE:-}" ]] && export TF_INKLING2_COMPUTE_LANE=1
ENV_ARGS=()
while IFS='=' read -r name _; do
  [[ "$name" == TENSORFOLD_API_KEY ]] && continue
  ENV_ARGS+=(-e "$name=${!name}")
done < <(env | grep -E '^(TENSORFOLD|TF_INKLING2|TF_NVFP4|TF_REASONING|TF_TOOL_CALL)_[A-Z0-9_]+=|^TF_KPROF(_[A-Z0-9_]+)?=' || true)
KEY_ARGS=(); [[ -z "${TENSORFOLD_API_KEY:-}" ]] || { export TENSORFOLD_API_KEY; KEY_ARGS=(-e TENSORFOLD_API_KEY); }

step 3 "Launch: container $CONTAINER_NAME"
log "tensorfold-native serve $MODEL_MOUNT --name $SERVED_NAME --host $HOST --port $PORT ${SERVE_ARGS[*]} ${USER_ARGS[*]}"
here_cmd=(docker run -d --name "$CONTAINER_NAME"
          --gpus all --ipc=host --network host
          --ulimit memlock=-1 --ulimit stack=67108864
          ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${KEY_ARGS[@]+"${KEY_ARGS[@]}"}
          -v "$MODEL_DIR":"$MODEL_MOUNT":ro
          ${DRAFT_MOUNT[@]+"${DRAFT_MOUNT[@]}"}
          -v "$KERNEL_CACHE/$KCACHE":/cache
          ${SESSION_MOUNT[@]+"${SESSION_MOUNT[@]}"}
          "$IMAGE"
          tensorfold-native serve "$MODEL_MOUNT" --name "$SERVED_NAME" --host "$HOST" --port "$PORT"
          "${SERVE_ARGS[@]}" ${USER_ARGS[@]+"${USER_ARGS[@]}"})
if (( DRY )); then
  printf '[dry-run]\n  %s\n' "$(printf '%q ' "${here_cmd[@]}" | sed 's/ $//')"
  exit 0
fi
"${here_cmd[@]}" >/dev/null

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap './stop.sh; exit 130' INT TERM
  docker logs -f "$CONTAINER_NAME" || true
  exit "$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)"
fi

step 4 "Loading: ~$WEIGHTS_GIB GiB of weights and the $(drafts_label) drafter"
NOISE='^\s*$|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|^\[tensorfold\] [^ ]+ "GET /(health|v1/models|models|metrics|stats|slots)[ ?]'
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
loaded_gib() {
  local pids
  pids=$(docker top "$CONTAINER_NAME" -eo pid 2>/dev/null | tail -n +2 | paste -sd'|')
  [[ -n "$pids" ]] || { echo 0; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v re="^($pids)$" '$1 ~ re { s += $2 } END { printf "%.1f", s / 1024 }'
}
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
start=$SECONDS
next_beat=15
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do
  running || fail "the server exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s; it is still running: docker logs -f $CONTAINER_NAME"
  a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if (( a < MEM_FLOOR_GIB )); then
    docker stop -t 5 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fail "only $a GiB memory left while loading (MEM_FLOOR_GIB $MEM_FLOOR_GIB): stopped"
  fi
  if (( SECONDS - start >= next_beat )); then
    printf '  %s⋯ %ss elapsed, %s GiB on the GPU (%s GiB free)%s\n' "$D" "$((SECONDS - start))" "$(loaded_gib)" "$a" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
LOADED_S=$((SECONDS - start))
log "Server answered after ${LOADED_S}s"

step 5 "Smoke test: one chat completion"
SERVED=$(served_name || echo "$SERVED_NAME")
# the model reasons first even at its lowest effort, so the smoke test gives it room and accepts reasoning or content
body="{\"model\": \"$SERVED\", \"max_tokens\": 256, \"temperature\": 0, \"chat_template_kwargs\": {\"reasoning_effort\": \"none\"}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
if smoke=$(curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); m = r["choices"][0]["message"]; c = (m.get("content") or m.get("reasoning") or m.get("reasoning_content") or "").strip(); assert c; print(repr(c[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed; the server is still running"
fi
if [[ "$RERISKED" == 1 ]]; then
  # the Derisked kit's readiness: a real completion with text in both the reasoning and the content fields
  body="{\"model\": \"$SERVED\", \"max_tokens\": 2048, \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 + 25? Answer in one sentence.\"}]}"
  if ready=$(curl -s --max-time 600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" |
             python3 -c 'import json,sys; m = json.load(sys.stdin)["choices"][0]["message"]; r = (m.get("reasoning") or "").strip(); c = (m.get("content") or "").strip(); assert r and c; print(len(r), "reasoning chars,", repr(c[:60]))' 2>/dev/null); then
    log "Derisked readiness OK (reasoning and content): $ready"
  else
    fail "Derisked readiness: no reply with both reasoning and content (deployment-kit/SERVE.md); the server is still running"
  fi
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Loaded   in ${LOADED_S}s · ${avail_gb} GiB was free
    Model    $MODEL_DIR ($QUANT_LABEL)
    Window   $(arg_value --context) tokens · $(arg_value --parallel) at once · $KV_DTYPE KV · drafts $(drafts_label)
    States   $( [[ "$PROMPT_CACHE_DISK" == 1 ]] && echo "RAM and $SESSION_DIR" || echo "RAM only" )
    Lane     $( [[ -n "${TF_INKLING2_COMPUTE_LANE:-}" && "${TF_INKLING2_COMPUTE_LANE}" != 0 ]] && echo "compute lane on (TF_INKLING2_COMPUTE_LANE=$TF_INKLING2_COMPUTE_LANE)" || echo "off" )
    Logs     docker logs -f $CONTAINER_NAME
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
