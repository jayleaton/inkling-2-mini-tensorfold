#!/usr/bin/env bash
# The Spark run for PRs #2-#4 and step 0, on dgx-02: one boot after another, each with its checks, a memory sampler and
# a log under ~/work/ab/<boot>. Usage: spark-run.sh BOOT... (ours-off ours-lane ours-toolfull ours-plain main)
set -uo pipefail
OUT=~/work/ab
mkdir -p "$OUT"
RUNNER_IMAGE=tensorfold-inkling2:c4-dc2ca8a-runner
MAIN_IMAGE=tensorfold-inkling2:main-920fe22

sampler() {  # MemAvailable every 2 s until the file $1.stop exists; peak = baseline available - lowest available
  local f=$1 base low
  base=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); low=$base
  while [[ ! -e "$f.stop" ]]; do
    local now; now=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    (( now < low )) && low=$now
    printf '%s %s\n' "$(date +%s)" "$now" >> "$f.samples"
    sleep 2
  done
  awk -v b="$base" -v l="$low" 'BEGIN { printf "baseline_available_gib=%.1f lowest_available_gib=%.1f peak_used_by_boot_gib=%.1f\n", b/1048576, l/1048576, (b-l)/1048576 }' > "$f"
}

boot() {  # boot NAME DIR ENV... -- starts the server with ENV in DIR, waiting until it answers
  local name=$1 dir=$2; shift 2
  local d=$OUT/$name; mkdir -p "$d"
  rm -f "$d/memory" "$d/memory.stop" "$d/memory.samples"
  sampler "$d/memory" & SAMPLER=$!
  ( cd "$dir" && env "$@" PREPARE=0 ./start.sh ) > "$d/start.log" 2>&1
  local rc=$?
  docker logs inkling2-mini-tf > "$d/engine-boot.log" 2>&1 || true
  return $rc
}

finish() {  # finish NAME: engine log, stop, memory
  local d=$OUT/$1
  docker logs inkling2-mini-tf > "$d/engine.log" 2>&1 || true
  ( cd ~/work/inkling-recipe && ./stop.sh ) >> "$d/start.log" 2>&1
  touch "$d/memory.stop"; wait "$SAMPLER" 2>/dev/null
  cat "$d/memory"
}

t() {  # t NAME LABEL CMD...: run a check in tools/, output to <boot>/<label>.txt
  local d=$OUT/$1 label=$2; shift 2
  echo "== $1 $label: $*"
  ( cd ~/work/inkling-recipe/tools && timeout 10800 "$@" ) > "$d/$label.txt" 2>&1
  echo "   exit $? : $(tail -1 "$d/$label.txt")"
}

LANE=(--tasks 20 --seed 1 --max-tokens 32768 --deadline-s 300)
for b in "$@"; do
  echo "#### $b $(date -Is)"
  case $b in
    main-2)
      boot "$b" ~/work/inkling-main IMAGE=$MAIN_IMAGE DRAFTS=0 KV_DTYPE=bf16 VISION=0 PROMPT_CACHE_DISK=0 || { echo "boot failed"; finish "$b"; continue; }
      t "$b" features ./features.py --no-vision
      t "$b" speed ./speed.py
      finish "$b" ;;
    ours-off-2)
      boot "$b" ~/work/inkling-recipe RUNNER=1 IMAGE=$RUNNER_IMAGE COMPUTE_LANE=0 || { echo "boot failed"; finish "$b"; continue; }
      t "$b" features ./features.py
      t "$b" speed ./speed.py
      t "$b" lane-P ./lane_ab.py P "${LANE[@]}" --out "$OUT/$b/lane-P.jsonl"
      t "$b" lane-D ./lane_ab.py D "${LANE[@]}" --out "$OUT/$b/lane-D.jsonl"
      finish "$b" ;;
    ours-lane-2)
      boot "$b" ~/work/inkling-recipe RUNNER=1 IMAGE=$RUNNER_IMAGE COMPUTE_LANE=1 || { echo "boot failed"; finish "$b"; continue; }
      t "$b" lane-L ./lane_ab.py L "${LANE[@]}" --out "$OUT/$b/lane-L.jsonl"
      finish "$b" ;;
    ours-toolfull-2)
      boot "$b" ~/work/inkling-recipe RUNNER=1 IMAGE=$RUNNER_IMAGE COMPUTE_LANE=tool-full || { echo "boot failed"; finish "$b"; continue; }
      t "$b" exact ./exact.py
      t "$b" features ./features.py --no-vision
      t "$b" lane-T ./lane_ab.py T "${LANE[@]}" --out "$OUT/$b/lane-T.jsonl"
      finish "$b" ;;
    *) echo "unknown boot $b" ;;
  esac
done
echo "#### done $(date -Is)"
