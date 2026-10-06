#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
export UV_CACHE_DIR="$root/build/uv-cache"
mkdir -p build
if [ ! -x .venv/bin/python ]; then uv venv --python 3.11 .venv; fi
uv pip sync --python .venv/bin/python requirements.lock \
    --index-url https://download.pytorch.org/whl/cpu
