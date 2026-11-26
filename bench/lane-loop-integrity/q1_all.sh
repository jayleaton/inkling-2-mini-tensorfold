#!/usr/bin/env bash
# Q1's boots in order on one Spark: each phase boots its engine (boot.sh), runs q1.py against it and stops it.
#   q1_all.sh <out dir> <phase>...     phases: S X S0 X-dspark S-off, or with the patched image (IMAGE):
#     X-sums S-sums   the lane cases with state sums
#     X-decode S-decode   each of that run's histories again: the prompt prefilled, the rest as plain decode rows
#     X-plant             case 1 with a visit's conv state bit-flipped (PLANT=<visit>@<position>[:bit])
#   X-disk S-disk     arm D again on that phase's arm A histories
# A phase's failed checks do not stop the next phase; every phase leaves <out>/<phase>/{rows.jsonl,ids/,log,exit}.
set -uo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$1
shift
mkdir -p "$out"
subset=1,4,7,10

boot() { # kernels drafter sessions [sums dir] [extra env]
  KERNELS=$1 DRAFTER=$2 SESSIONS=$3 SUMS=${4:-} EXTRA_ENV=${5:-} ARM=L RUN_DIR="$dir" "$kit/boot.sh" >> "$dir/log" 2>&1
}

for phase in "$@"; do
  dir="$out/$phase"
  mkdir -p "$dir"
  sessions="$dir/sessions"
  status=0
  case "$phase" in
    S|X)
      boot "$phase" dflash2 "$sessions" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" lane --out "$dir" >> "$dir/log" 2>&1 || status=1
      python3 "$kit/q1.py" control --out "$dir" >> "$dir/log" 2>&1 || status=1
      python3 "$kit/q1.py" disk --out "$dir" --cases 2,7 --samplers greedy,t07 --sessions "$sessions" \
        --restart "KERNELS=$phase DRAFTER=dflash2 SESSIONS=$sessions ARM=L RUN_DIR=$dir $kit/boot.sh" >> "$dir/log" 2>&1 || status=1
      ;;
    X-disk|S-disk)
      k=${phase%%-*}
      mkdir -p "$dir/ids"
      cp "$out/$k/ids/"[0-9]*.json "$dir/ids/"
      boot "$k" dflash2 "$sessions" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" disk --out "$dir" --cases 2,7 --samplers greedy,t07 --sessions "$sessions" \
        --restart "KERNELS=$k DRAFTER=dflash2 SESSIONS=$sessions ARM=L RUN_DIR=$dir $kit/boot.sh" >> "$dir/log" 2>&1 || status=1
      ;;
    S0)
      boot S0 dflash2 "$sessions" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" lane --light --out "$dir" >> "$dir/log" 2>&1 || status=1
      python3 "$kit/q1.py" control --out "$dir" >> "$dir/log" 2>&1 || status=1
      ;;
    X-dspark|S-dspark)
      boot "${phase%%-*}" dspark "$sessions" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" lane --light --cases "$subset" --samplers greedy,t07 --out "$dir" >> "$dir/log" 2>&1 || status=1
      ;;
    X-off|S-off)
      boot "${phase%%-*}" off "$sessions" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" lane --light --samplers greedy,t07 --out "$dir" >> "$dir/log" 2>&1 || status=1
      ;;
    S-sums|X-sums|S0-sums)
      boot "${phase%%-*}" dflash2 "$sessions" "$dir/sums" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" lane --light --samplers greedy,t07 --sums "$dir/sums/sums.jsonl" --out "$dir" >> "$dir/log" 2>&1 || status=1
      ;;
    X-decode|S-decode)
      mkdir -p "$dir/sums"
      echo 1000000000 > "$dir/sums/decode_from"
      boot "${phase%%-*}" dflash2 "$sessions" "$dir/sums" \
        "TF_INKLING2_STATE_SUMS_DECODE_PROMPT=8 TF_INKLING2_STATE_SUMS_DECODE_FROM=/sums/decode_from" ||
        { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" ref --from "$out/${phase%%-*}-sums" --samplers greedy,t07 --sums "$dir/sums/sums.jsonl" \
        --decode-from "$dir/sums/decode_from" --out "$dir" >> "$dir/log" 2>&1 || status=1
      ;;
    X-plant)
      boot X dflash2 "$sessions" "$dir/sums" "TF_INKLING2_STATE_SUMS_PLANT=${PLANT:?}" || { echo "boot failed" > "$dir/exit"; continue; }
      python3 "$kit/q1.py" lane --light --cases 1 --samplers greedy --sums "$dir/sums/sums.jsonl" --out "$dir" >> "$dir/log" 2>&1 || status=1
      ;;
    *) echo "unknown phase $phase" >&2; exit 2 ;;
  esac
  RUN_DIR="$dir" "$kit/boot.sh" stop
  echo "$status" > "$dir/exit"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/done.$(printf '%s' "$*" | tr ' ' '+')"
