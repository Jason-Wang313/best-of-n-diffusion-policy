#!/usr/bin/env bash
set -euo pipefail

if command -v python.exe >/dev/null 2>&1; then
  PY=(python.exe)
elif command -v python >/dev/null 2>&1; then
  PY=(python)
elif command -v py >/dev/null 2>&1; then
  PY=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  echo "No Python interpreter found on PATH" >&2
  exit 127
fi

"${PY[@]}" scripts/claim_audit.py --fail-on-error "$@"
