#!/usr/bin/env bash
# Q2's blocks on one Spark: a block boots each arm in turn (the order rotates by block), asks its 12 questions and stops.
#   q2_all.sh <out dir> <spark name> <private sets dir> <block>...
# Stops between blocks when <out dir>/STOP exists. A block is complete when <out dir>/block-N.done exists.
set -uo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$1
spark=$2
private=$3
shift 3
mkdir -p "$out"
orders=("P L T" "L T P" "T P L")

for block in "$@"; do
  [[ -e "$out/STOP" ]] && break
  [[ -e "$out/block-$block.done" ]] && continue
  ok=1
  for arm in ${orders[$((block % 3))]}; do
    if ! ARM=$arm KERNELS=S RUN_DIR="$out/boots" "$kit/boot.sh" >> "$out/log" 2>&1; then ok=0; break; fi
    python3 "$kit/q2.py" run --arm "$arm" --block "$block" --sets "$kit/sets,$private" --spark "$spark" \
      --out "$out/rows-$spark.jsonl" >> "$out/log" 2>&1 || ok=0
    docker logs lli-engine > "$out/boots/engine-b$block-$arm.log" 2>&1 || true
    "$kit/boot.sh" stop
  done
  [[ $ok == 1 ]] && date -u +%Y-%m-%dT%H:%M:%SZ > "$out/block-$block.done"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/finished"
