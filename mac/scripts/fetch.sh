#!/usr/bin/env bash
# Download upstream Inkling-2-Mini (thinkingmachines/Inkling-2-Mini) or DFlash2 drafter
# (incoai/Inkling-2-Mini-DFlash2) at pinned revisions. Resumable via curl -C -.
# Usage: mac/scripts/fetch.sh [DIR]  |  mac/scripts/fetch.sh drafter [DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

if [[ "${1:-}" == drafter ]]; then
  DIR="${2:-$DRAFTER_DIR}"
  REPO="$DRAFTER_REPO"
  REV="$DRAFTER_REV"
  FILES=(config.json README.md "$DRAFTER_NAME")
else
  DIR="${1:-$MODEL_DIR}"
  REPO="$HF_REPO"
  REV="$HF_REVISION"
  FILES=(config.json generation_config.json hf_quant_config.json chat_template.jinja processor_config.json
         special_tokens_map.json tokenizer.json tokenizer_config.json tiktoken/tokenizer.model
         model.safetensors.index.json README.md "$MTP_NAME" "$SHARD_NAME")
fi

AUTH=(); [[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$DIR"
for f in "${FILES[@]}"; do
  mkdir -p "$DIR/$(dirname "$f")"
  log "fetching $f"
  curl -fL --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$DIR/$f" "https://huggingface.co/$REPO/resolve/$REV/$f" \
    || die "could not fetch $f from $REPO@${REV:0:12}"
done
log "$REPO@${REV:0:12} is in $DIR"
