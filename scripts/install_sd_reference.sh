#!/usr/bin/env bash
# Independent project-local CPU reference; no changes to existing LLM packages.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export UV_CACHE_DIR="$PWD/build/cache/uv"
export UV_LINK_MODE=copy
uv venv build/venvs/sd-image --python .venv/bin/python --allow-existing
uv pip install --python build/venvs/sd-image/bin/python --only-binary :all: --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
uv pip install --python build/venvs/sd-image/bin/python --only-binary :all: diffusers==0.36.0 transformers==4.57.6 numpy==2.4.6 safetensors==0.7.0 huggingface-hub==0.36.2 Pillow==12.0.0
uv pip freeze --python build/venvs/sd-image/bin/python > build/sd-reference-packages.txt
