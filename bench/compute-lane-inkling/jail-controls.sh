#!/usr/bin/env bash
# HumanEval/reference jail qualification in this Spark's bound runner image (CPU only; docker run --network none).
# Canonical passes, wrong fails, infinite loop times out, os._exit(0) fails, plus six reference-executor cases.
#
# Usage: ./jail-controls.sh FREEZE HOST OUT_DIR
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
[[ $# -eq 3 ]] || { sed -n '2,5s/^# \{0,1\}//p' "$0"; exit 2; }
python3 -I ink/jail_ink.py qualify --freeze "$1" --host "$2" --out "$3"
