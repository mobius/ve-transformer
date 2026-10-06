#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
    exec python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised "$@"
fi
shift
# Preserve positional modes while forwarding backend and mode options to argparse.
if [[ $# -gt 0 && $1 != --* ]]; then
    mode=$1
    shift
    set -- --mode "$mode" "$@"
fi
bash scripts/check_environment.sh
.venv/bin/python tests/check_qwen_model.py "$@"
