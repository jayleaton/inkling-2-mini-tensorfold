#!/usr/bin/env bash
# Prepare weights (fetch + SHA-256 verify) and build TensorFold Metal native server + libtfgrammar.dylib.
# Usage: mac/scripts/prepare.sh [--rebuild] [--rehash]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

REBUILD=0
REHASH=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    --rehash) REHASH=1 ;;
    *) die "unknown argument: $arg (use --rebuild or --rehash)" ;;
  esac
done

for cmd in zig git curl python3 shasum xcrun; do
  command -v "$cmd" >/dev/null 2>&1 || die "$cmd is required on macOS"
done
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "mac/ recipe targets Apple Silicon macOS (arm64)"

mkdir -p "$BUILD_ROOT" "$STATE_DIR" "$LOG_DIR"

# 1. Fetch weights if missing
if [[ "$MODEL_FETCH" == 1 && ( ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/$SHARD_NAME" || ! -f "$MODEL_DIR/$MTP_NAME" ) ]]; then
  log "Fetching $HF_REPO@${HF_REVISION:0:12} into $MODEL_DIR"
  ./scripts/fetch.sh "$MODEL_DIR"
fi
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/$SHARD_NAME" && -f "$MODEL_DIR/$MTP_NAME" ]] || \
  die "$MODEL_DIR is missing checkpoint files (config.json, $SHARD_NAME, $MTP_NAME)"

if [[ "$DFLASH2" == 1 ]]; then
  if [[ "$DRAFTER_FETCH" == 1 && ( ! -f "$DRAFTER_DIR/config.json" || ! -f "$DRAFTER_DIR/$DRAFTER_NAME" ) ]]; then
    log "Fetching $DRAFTER_REPO@${DRAFTER_REV:0:12} into $DRAFTER_DIR"
    ./scripts/fetch.sh drafter "$DRAFTER_DIR"
  fi
  [[ -f "$DRAFTER_DIR/config.json" && -f "$DRAFTER_DIR/$DRAFTER_NAME" ]] || \
    die "$DRAFTER_DIR is missing DFlash2 files (config.json, $DRAFTER_NAME)"
fi

# 2. SHA-256 verify weights when changed or --rehash
stamp=$(weights_stamp)
if (( REHASH )) || [[ "$(cat "$VERIFIED_MARKER" 2>/dev/null || true)" != "$stamp" ]]; then
  verify_sha() {
    local file=$1 want=$2 got
    log "Verifying SHA-256 of $(basename "$file")"
    got=$(shasum -a 256 "$file" | awk '{print $1}')
    [[ "$got" == "$want" ]] || die "SHA-256 mismatch for $file: got $got, expected $want"
  }
  verify_sha "$MODEL_DIR/$SHARD_NAME" "$SHARD_SHA256"
  verify_sha "$MODEL_DIR/$MTP_NAME" "$MTP_SHA256"
  if [[ "$DFLASH2" == 1 ]]; then
    verify_sha "$DRAFTER_DIR/$DRAFTER_NAME" "$DRAFTER_SHA256"
  fi
  printf '%s\n' "$stamp" > "$VERIFIED_MARKER"
  log "All weight SHA-256 checksums verified"
else
  log "Weights already SHA-256 verified"
fi

# 3. Clone/prepare xgrammar at pinned tag
if [[ ! -d "$XGRAMMAR_DIR/.git" ]]; then
  log "Cloning xgrammar $XGRAMMAR_TAG into $XGRAMMAR_DIR"
  git clone --depth 1 --branch "$XGRAMMAR_TAG" --recurse-submodules --shallow-submodules "$XGRAMMAR_REPO" "$XGRAMMAR_DIR"
fi

# 4. Clone/patch/build TensorFold Metal native + grammar
want_hash=$(build_hash)
have_hash=$(cat "$BUILD_ROOT/.build_hash" 2>/dev/null || true)
if (( REBUILD )) || [[ "$have_hash" != "$want_hash" || ! -x "$BIN_PATH" || ! -f "$GRAMMAR_LIB" ]]; then
  log "Building TensorFold Metal ($TF_REF, patches $want_hash)"
  if [[ ! -d "$TF_SRC_DIR/.git" ]]; then
    if [[ -n "$TF_LOCAL_REF" && -d "$TF_LOCAL_REF/.git" ]]; then
      git clone --shared "$TF_LOCAL_REF" "$TF_SRC_DIR"
    else
      git clone "$TF_REPO" "$TF_SRC_DIR"
    fi
  fi
  git -C "$TF_SRC_DIR" reset --hard -q
  git -C "$TF_SRC_DIR" clean -fdx -e zig-out -e .zig-cache -q
  if ! git -C "$TF_SRC_DIR" cat-file -e "${TF_REF}^{commit}" 2>/dev/null; then
    git -C "$TF_SRC_DIR" fetch origin "$TF_REF" --depth=1 || git -C "$TF_SRC_DIR" fetch origin
  fi
  git -C "$TF_SRC_DIR" checkout -q "$TF_REF"
  for p in patches/*.patch; do
    [[ -f "$p" ]] || continue
    log "Applying $(basename "$p")"
    git -C "$TF_SRC_DIR" apply "$(pwd)/$p"
  done
  ( cd "$TF_SRC_DIR" && zig build native grammar -Doptimize=ReleaseFast -Dxgrammar="$XGRAMMAR_DIR" )
  printf '%s\n' "$want_hash" > "$BUILD_ROOT/.build_hash"
  log "Built $BIN_PATH and $GRAMMAR_LIB"
else
  log "TensorFold Metal build is up to date ($want_hash)"
fi

prepared_state > "$PREPARED_MARKER"
log "Preparation complete"
