#!/usr/bin/env bash
# Launch the Local AI Helper desktop window (starts the Docker stack if needed).
set -euo pipefail
cd "$(dirname "$0")"
 
# Prefer a newer Python than macOS's built-in 3.9 if one is installed (e.g. brew install python@3.12).
PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
[ -n "$PY" ] || { echo "Python 3 not found. Install it with: brew install python@3.12"; exit 1; }
 
# (Re)build the virtualenv unless it already works.
if ! .venv/bin/python -c "import webview" >/dev/null 2>&1; then
  echo "Setting up the desktop environment with $PY ..."
  rm -rf .venv
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip   # old pip cannot see modern macOS wheels
  # --prefer-binary: use prebuilt wheels (so pyobjc is never compiled) but still allow
  # the tiny pure-Python packages that only ship as source (proxy_tools).
  .venv/bin/python -m pip install --prefer-binary -r desktop/requirements.txt \
    || { echo "Install failed. Remove .venv and try again; if it persists, install Python 3.12 (brew install python@3.12)."; rm -rf .venv; exit 1; }
fi
 
# .env is optional (all settings have defaults); create it from the example if that file exists.
if [ ! -f .env ]; then
  if [ -f .env.example ]; then cp .env.example .env; else : > .env; fi
fi
exec .venv/bin/python desktop/helper.py
 