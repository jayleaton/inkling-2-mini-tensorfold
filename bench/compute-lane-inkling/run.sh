#!/usr/bin/env bash
# Run frozen turns on this Spark, one after another, each through ink/controller.py (three boots per turn).
#
# Usage: ./run.sh FREEZE HOST PASS [TURN...] [-- controller flags]
#   ./run.sh ~/work/ink-runs/freeze spark1 c8k                  # all eight turns: s01..s06, then c01, c02
#   ./run.sh ~/work/ink-runs/freeze spark1 c32k c01 c02         # compute-DL only
#   ./run.sh ~/work/ink-runs/freeze spark1 c8k s03 -- --resume  # continue a STOPPED boot in a fresh boot
# RUNS (default ~/work/ink-runs/runs) is where every attempt is written; a QUALIFIED boot is never rerun.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
[[ $# -ge 3 ]] || { sed -n '2,8s/^# \{0,1\}//p' "$0"; exit 2; }
FREEZE=$1 HOST=$2 PASS=$3
shift 3
TURNS=()
while [[ $# -gt 0 && "$1" != -- ]]; do TURNS+=("$1"); shift; done
[[ "${1:-}" == -- ]] && shift
[[ ${#TURNS[@]} -gt 0 ]] || TURNS=(s01 s02 s03 s04 s05 s06 c01 c02)
RUNS=${RUNS:-$HOME/work/ink-runs/runs}
for turn in "${TURNS[@]}"; do
  echo "== $PASS $turn ($(date -u +%FT%TZ))"
  python3 -I ink/controller.py run --freeze "$FREEZE" --host "$HOST" --pass "$PASS" --turn "$turn" \
    --runs "$RUNS" "$@"
done
