#!/usr/bin/env bash
# CPU checks: byte-compile every .py, bash -n and shellcheck every .sh, then ink/selftest.py (fake server, no model).
#
# Usage: ./selftest.sh WORK_DIR [--quick] [--jail-image TAG]   (WORK_DIR outside this folder, its path holding "selftest")
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
[[ $# -ge 1 ]] || { sed -n '2,4s/^# \{0,1\}//p' "$0"; exit 2; }
find . -name '*.py' -not -path '*/__pycache__/*' -print0 | sort -z | while IFS= read -r -d '' f; do
  python3 -I -m py_compile "$f"
done
echo "py_compile: ok ($(find . -name '*.py' -not -path '*/__pycache__/*' | wc -l) files)"
for f in ./*.sh; do bash -n "$f"; done
if command -v shellcheck >/dev/null; then shellcheck ./*.sh; echo "shellcheck: ok"; else echo "shellcheck: not installed"; fi
python3 -I ink/selftest.py --work "$@"
