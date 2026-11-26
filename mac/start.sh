#!/usr/bin/env bash
# Serve thinkingmachines/Inkling-2-Mini natively on Apple Silicon (Metal) using TensorFold's Zig engine.
# Fetches + SHA-verifies weights, builds tensorfold-native + libtfgrammar.dylib when needed,
# starts the OpenAI-compatible API server on PORT (default 8888), waits for /health, and runs a smoke test.
# Stop it with mac/stop.sh.
#
# Usage: mac/start.sh [restart] [extra tensorfold-native serve args]
#   mac/start.sh                          # 2 streams, 128k context, DFlash2 drafter, vision on
#   mac/start.sh restart                  # stop, then start again
#   DRAFTER=dspark mac/start.sh restart   # use the checkpoint's built-in DSpark drafter
#   DRAFTS=0 mac/start.sh restart         # disable speculative drafting
#   VISION=0 mac/start.sh restart         # text-only (skip vision tower)
#   FOREGROUND=1 mac/start.sh             # run in foreground
#   DRY_RUN=1 mac/start.sh                # print the serve command and exit
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT must be 1..1048576, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL must be 1..16, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS must be a positive integer, not $MAX_TOKENS"
for v in THINKING DRAFTS VISION VISION_URLS; do [[ "${!v}" =~ ^[01]$ ]] || die "$v must be 0 or 1, not ${!v}"; done
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
if [[ "$DRAFTS" != 1 ]]; then SERVE_ARGS+=(--no-drafts)
elif [[ "$DRAFTER" == dflash2 ]]; then SERVE_ARGS+=(--drafter "$DRAFTER_DIR")
else SERVE_ARGS+=(--drafter "$MODEL_DIR"); fi
if [[ "$VISION" == 1 ]]; then
  SERVE_ARGS+=(--vision)
  [[ "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
  [[ -z "$VISION_MAX_IMAGES" ]] || SERVE_ARGS+=(--vision-max-images "$VISION_MAX_IMAGES")
  [[ -z "$VISION_IMAGE_TOKENS" ]] || SERVE_ARGS+=(--vision-image-tokens "$VISION_IMAGE_TOKENS")
fi
[[ -z "$PROMPT_CACHE_GIB" ]] || SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
USER_ARGS=("$@")

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

mkdir -p "$BUILD_ROOT" "$STATE_DIR" "$LOG_DIR"

if (( ! DRY )) && [[ "$MODE" == start ]] && running; then
  log "Already running (pid $(cat "$PID_FILE"), model: $(served_name || echo "loading"), port $PORT)"
  log "Use mac/start.sh restart to restart, or mac/stop.sh to stop."
  exit 0
fi

# Step 1: Prepare weights and Metal build
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null || true)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: mac/scripts/prepare.sh would run now"
  else ./scripts/prepare.sh; fi
else
  log "Ready: $BIN_PATH ($(build_hash)) and $MODEL_DIR"
fi

# Step 2: Stop previous instance on restart, check port
if (( ! DRY )) && running; then
  ./stop.sh
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if (( DRY )); then warn "DRY_RUN: port $PORT is in use"
  else die "port $PORT is already in use"; fi
fi

CMD=("$BIN_PATH" serve "$MODEL_DIR" --name "$SERVED_NAME" --host "$HOST" --port "$PORT"
     "${SERVE_ARGS[@]}" ${USER_ARGS[@]+"${USER_ARGS[@]}"})

if (( DRY )); then
  printf '[dry-run]\n  %s\n' "$(printf '%q ' "${CMD[@]}" | sed 's/ $//')"
  exit 0
fi

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  exec "${CMD[@]}"
fi

# Step 3: Launch in background
LOG_FILE="$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"
log "Starting tensorfold-native on $HOST:$PORT (drafts: $(drafts_label), context: $CONTEXT, parallel: $PARALLEL)"
nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"

# Step 4: Wait for /health
start=$SECONDS
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    tail -n 40 "$LOG_FILE" >&2 || true
    die "tensorfold-native exited before becoming ready (see $LOG_FILE)"
  fi
  (( SECONDS - start < WAIT_TIMEOUT )) || die "not ready after ${WAIT_TIMEOUT}s (pid $SERVER_PID, see $LOG_FILE)"
  sleep 1
done
LOADED_S=$((SECONDS - start))
log "Server ready in ${LOADED_S}s (pid $SERVER_PID)"

# Step 5: Smoke test
SERVED=$(served_name || echo "$SERVED_NAME")
body="{\"model\": \"$SERVED\", \"max_tokens\": 128, \"temperature\": 0, \"chat_template_kwargs\": {\"reasoning_effort\": \"none\"}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
if smoke=$(curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); m = r["choices"][0]["message"]; c = (m.get("content") or m.get("reasoning") or m.get("reasoning_content") or "").strip(); assert c; print(repr(c[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "Smoke test OK: $smoke"
else
  tail -n 30 "$LOG_FILE" >&2 || true
  die "smoke test failed; server is still running (pid $SERVER_PID)"
fi

log "LIVE: http://$API_HOST:$PORT/v1 (model: $SERVED, logs: $LOG_FILE)"
