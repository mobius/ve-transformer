#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ ! -x .venv/bin/python ]; then bash scripts/setup_env.sh; fi
export UV_CACHE_DIR="$root/build/uv-cache"
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements-llm.lock
