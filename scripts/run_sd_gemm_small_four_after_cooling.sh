#!/usr/bin/env bash
set -euo pipefail
[[ ${VE_TRANSFORMER_TEMPERATURE_SUPERVISED:-0} == 1 ]] || exit 2
bash scripts/check_environment.sh
.venv/bin/python scripts/wait_thermal_ready.py --output build/gemm-small-model-four-retry-launch-cooling.json
exec .venv/bin/python scripts/sample_ve_memory.py --timeout 600 "$@"
