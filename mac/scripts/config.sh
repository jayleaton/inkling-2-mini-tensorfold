#!/usr/bin/env bash
# shellcheck disable=SC2034
# Shared settings for mac/start.sh, mac/stop.sh and mac/scripts/*.sh.
# Precedence: environment > mac/.env (KEY=value lines, read, never executed) > defaults below.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# Checkpoint: upstream thinkingmachines/Inkling-2-Mini at pinned revision 7a67ae23.
MODEL_DIR="${MODEL_DIR:-$HOME/models/hf/Inkling-2-Mini}"
MODEL_LABEL="${MODEL_LABEL:-Inkling-2-Mini}"
HF_REPO="${HF_REPO:-thinkingmachines/Inkling-2-Mini}"
HF_REVISION="${HF_REVISION:-7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6}"
SHARD_NAME="model-00001-of-00001.safetensors"
SHARD_SHA256="${SHARD_SHA256:-65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1}"
MTP_NAME="mtp.safetensors"
MTP_SHA256="${MTP_SHA256:-7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2}"
MODEL_FETCH="${MODEL_FETCH:-1}"
QUANT_LABEL="ModelOpt NVFP4 experts, MXFP8 layers 20-23, bf16 MTP"

# Engine: upstream ashhart/TensorFold main pinned to 920fe22c9 + mac/patches/*.patch
# (#4622 Metal drafter window fit, #4917 native Metal DFlash2 drafter).
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-920fe22c97995d3dae9bed3e320f08aaa64c25ae}"
TF_LOCAL_REF="${TF_LOCAL_REF:-}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
XGRAMMAR_REPO="${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar}"
XGRAMMAR_TAG="${XGRAMMAR_TAG:-v0.2.8}"

# Serving defaults for Mac Studio (Metal): 2 streams, 128k context, DFlash2 drafter, vision on.
SERVED_NAME="${SERVED_NAME:-$MODEL_LABEL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CONTEXT="${CONTEXT:-131072}"
PARALLEL="${PARALLEL:-2}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
DRAFTS="${DRAFTS:-1}"
DRAFTER="${DRAFTER:-dflash2}"
[[ "$DRAFTER" =~ ^(dflash2|dspark)$ ]] || { echo "DRAFTER is dflash2 or dspark, not $DRAFTER" >&2; exit 1; }
DRAFTER_DIR="${DRAFTER_DIR:-$HOME/models/hf/Inkling-2-Mini-DFlash2}"
DRAFTER_REPO="${DRAFTER_REPO:-incoai/Inkling-2-Mini-DFlash2}"
DRAFTER_REV="${DRAFTER_REV:-4f9088ea66c21f2b99ed657d1941c306b03e84d2}"
DRAFTER_NAME="model.safetensors"
DRAFTER_SHA256="${DRAFTER_SHA256:-217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee}"
DRAFTER_FETCH="${DRAFTER_FETCH:-1}"
DFLASH2=0; [[ "$DRAFTS" == 1 && "$DRAFTER" == dflash2 ]] && DFLASH2=1

drafts_label() { if [[ "$DRAFTS" != 1 ]]; then echo off; elif [[ "$DRAFTER" == dflash2 ]]; then echo DFlash2; else echo DSpark; fi; }

VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-}"
THINKING="${THINKING:-1}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"

BUILD_ROOT="${BUILD_ROOT:-$HOME/.cache/tensorfold-inkling2-mac}"
TF_SRC_DIR="$BUILD_ROOT/tensorfold"
XGRAMMAR_DIR="$BUILD_ROOT/xgrammar"
BIN_PATH="$TF_SRC_DIR/zig-out/native/bin/tensorfold-native"
GRAMMAR_LIB="$TF_SRC_DIR/zig-out/native/lib/libtfgrammar.dylib"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/inkling2-mini-tf-mac}"
LOG_DIR="${LOG_DIR:-$BUILD_ROOT/logs}"
PID_FILE="$STATE_DIR/server.pid"
PREPARED_MARKER="$STATE_DIR/prepared"
VERIFIED_MARKER="$STATE_DIR/verified"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

build_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null || true
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION xgrammar-$XGRAMMAR_TAG"
  ) | shasum -a 256 | cut -c1-12
}

weights_stamp() {
  local f out=""
  local -a files=("$MODEL_DIR/$SHARD_NAME" "$MODEL_DIR/$MTP_NAME")
  [[ "$DFLASH2" == 1 ]] && files+=("$DRAFTER_DIR/$DRAFTER_NAME")
  for f in "${files[@]}"; do
    out+="$(stat -f '%z %m' "$f" 2>/dev/null || echo missing) "
  done
  echo "$out"
}

prepared_state() {
  local bin_ok=missing
  [[ -x "$BIN_PATH" && -f "$GRAMMAR_LIB" ]] && bin_ok=present
  echo "model=$MODEL_DIR $(weights_stamp)bin=$bin_ok build=$(build_hash)"
}
