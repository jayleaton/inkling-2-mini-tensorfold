#!/usr/bin/env bash
# Boots one engine container for the lane loop integrity test on a Spark, from the recipe's own `docker run` line
# (DRY_RUN=1 ./start.sh, 2026-11-26), and waits for /health. Prints the exact line it ran.
#   ARM=P|L|T            lane off, compute lane, tool-full                      (default L)
#   KERNELS=S|S0|X       shipped (fp8 KV, FP8 attention, tensor-core prefill), S with PREFILL_TC=0, exact bf16 (default S)
#   DRAFTER=dflash2|dspark|off                                                  (default dflash2)
#   IMAGE, NAME, PORT, CONTEXT, PARALLEL, SESSIONS (kept across boots when set), SUMS (a host folder: state sums on),
#   EXTRA_ENV ("K=V K=V"), RUN_DIR (where the line and the engine's facts are written)
# stop: boot.sh stop   (only the container named NAME)
set -euo pipefail

ARM="${ARM:-L}"
KERNELS="${KERNELS:-S}"
DRAFTER="${DRAFTER:-dflash2}"
IMAGE="${IMAGE:-tensorfold-inkling2:c4-dc2ca8a-runner}"
NAME="${NAME:-lli-engine}"
PORT="${PORT:-8888}"
CONTEXT="${CONTEXT:-131072}"
PARALLEL="${PARALLEL:-2}"
HOME_DIR="${HOME_DIR:-$(getent passwd 1000 | cut -d: -f6)}"
WORK="${WORK:-$HOME_DIR/work/lli}"
RUN_DIR="${RUN_DIR:-$WORK/boots}"
KCACHE="${KCACHE:-$HOME_DIR/.cache/tensorfold-inkling2/43e0fb792314}"

stop() {
  docker inspect "$NAME" >/dev/null 2>&1 || return 0
  docker stop -t 30 "$NAME" >/dev/null
  mkdir -p "$RUN_DIR"
  docker logs "$NAME" > "$RUN_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$NAME.log" 2>&1 || true
  docker rm "$NAME" >/dev/null
}

if [[ "${1:-}" == stop ]]; then stop; exit 0; fi

env_args=(-e TENSORFOLD_MEMORY_RESERVE_GIB=9)
kv=fp8
case "$KERNELS" in
  S)  env_args+=(-e TF_INKLING2_ATTN_FP8=1 -e TF_INKLING2_PREFILL_TC=1) ;;
  S0) env_args+=(-e TF_INKLING2_ATTN_FP8=1 -e TF_INKLING2_PREFILL_TC=0) ;;
  X)  env_args+=(-e TF_INKLING2_ATTN_FP8=0 -e TF_INKLING2_PREFILL_TC=0); kv=bf16 ;;
  *) echo "KERNELS is S, S0 or X" >&2; exit 2 ;;
esac
runner=/opt/tensorfold/share/runner
case "$ARM" in
  P) ;;
  L) env_args+=(-e TF_INKLING2_COMPUTE_LANE=1) ;;
  T) env_args+=(-e TF_INKLING2_COMPUTE_LANE=1 -e TF_INKLING2_RUNNER=1
       -e "TF_INKLING2_RUNNER_CIRCUIT=$runner/calc.tfpc" -e "TF_INKLING2_RUNNER_FORMS=fence,repl,inline,trace"
       -e TF_INKLING2_RUNNER_BACKEND=graph -e "TF_INKLING2_RUNNER_GRAPH_BANK=$runner/graph-v7-powmod/bank/bank.json"
       -e TF_INKLING2_RUNNER_STATED=0 -e TF_INKLING2_RUNNER_INLINE_CHAIN=0 -e TF_INKLING2_RUNNER_INLINE_MARK=0
       -e TF_INKLING2_RUNNER_SYNC_GRAPH=0 -e TF_INKLING2_RUNNER_COMPILE_STEPS=10522 -e TF_INKLING2_RUNNER_CELLS=1
       -e TF_INKLING2_RUNNER_COMMIT=0 -e TF_INKLING2_COMPUTE_LANE_RUNNER=1
       -e "TF_INKLING2_COMPUTE_LANE_RUNNER_CIRCUIT=$runner/calc.tfpc" -e TF_INKLING2_COMPUTE_LANE_RUNNER_BACKEND=graph
       -e "TF_INKLING2_COMPUTE_LANE_RUNNER_GRAPH_BANK=$runner/graph-v7-powmod/bank/bank.json"
       -e TF_INKLING2_COMPUTE_LANE_RUNNER_SETTLE_US=3000) ;;
  *) echo "ARM is P, L or T" >&2; exit 2 ;;
esac
for kvp in ${EXTRA_ENV:-}; do env_args+=(-e "$kvp"); done

mkdir -p "$RUN_DIR"
sessions="${SESSIONS:-}"
if [[ -z "$sessions" ]]; then sessions=$(mktemp -d "$WORK/sessions.XXXXXX"); fi
mkdir -p "$sessions"
vol_args=(-v "$HOME_DIR/models/hf/Inkling-2-Mini:/models/ink:ro" -v "$KCACHE:/cache" -v "$sessions:/sessions")
drafter_args=()
case "$DRAFTER" in
  dflash2) vol_args+=(-v "$HOME_DIR/models/hf/Inkling-2-Mini-DFlash2:/models/draft:ro"); drafter_args=(--drafter /models/draft) ;;
  dspark)  drafter_args=(--drafter /models/ink) ;;
  off)     drafter_args=(--no-drafts) ;;
  *) echo "DRAFTER is dflash2, dspark or off" >&2; exit 2 ;;
esac
if [[ -n "${SUMS:-}" ]]; then
  mkdir -p "$SUMS"
  vol_args+=(-v "$SUMS:/sums")
  env_args+=(-e TF_INKLING2_STATE_SUMS=/sums/sums.jsonl)
fi

stop
line=(docker run -d --name "$NAME" --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864
  "${env_args[@]}" "${vol_args[@]}" "$IMAGE" tensorfold-native serve /models/ink --name Inkling-2-Mini --host 0.0.0.0
  --port "$PORT" --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens 32768 --temperature 1.0 --top-p 1.0 --top-k 0
  --thinking "${drafter_args[@]}" --vision --kv-cache "$kv" --prompt-cache-disk /sessions)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
printf '%q ' "${line[@]}" > "$RUN_DIR/$stamp-$NAME.run"
printf '\n' >> "$RUN_DIR/$stamp-$NAME.run"
"${line[@]}" >/dev/null
for _ in $(seq 1 240); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    {
      echo "image $(docker inspect -f '{{.Image}}' "$NAME")"
      echo "arm $ARM kernels $KERNELS drafter $DRAFTER sessions $sessions"
      docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" | grep -E '^(TF_|TENSORFOLD_)' | sort
    } > "$RUN_DIR/$stamp-$NAME.facts"
    cat "$RUN_DIR/$stamp-$NAME.run" "$RUN_DIR/$stamp-$NAME.facts"
    exit 0
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]]; then
    docker logs --tail 40 "$NAME" >&2 || true
    echo "the engine stopped while starting" >&2
    exit 1
  fi
  sleep 2
done
echo "no /health after 480 s" >&2
exit 1
