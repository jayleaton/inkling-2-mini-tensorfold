#!/usr/bin/env bash
# Freeze the rerun and bind this Spark's image to it (no model requests).
#
# Usage: ./freeze.sh create FREEZE_DIR    # deterministic: the same tree gives the same root hash on every host
#        ./freeze.sh bind FREEZE_DIR HOST  # docker image inspect of the runner image; tf.patches must match
#        ./freeze.sh verify FREEZE_DIR     # inventory, harness and recipe drift
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
RECIPE=$(readlink -f ../..)
case "${1:-}" in
  create) [[ $# -eq 2 ]] || exit 2; python3 -I ink/freeze.py create --recipe "$RECIPE" --out "$2" ;;
  bind) [[ $# -eq 3 ]] || exit 2; python3 -I ink/freeze.py bind-host --freeze "$2" --host "$3" ;;
  verify) [[ $# -eq 2 ]] || exit 2; python3 -I ink/freeze.py verify --freeze "$2" --recipe "$RECIPE" ;;
  *) sed -n '2,6s/^# \{0,1\}//p' "$0"; exit 2 ;;
esac
