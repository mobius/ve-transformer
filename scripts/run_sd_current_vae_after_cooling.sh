#!/usr/bin/env bash
set -euo pipefail
[[ ${VE_TRANSFORMER_TEMPERATURE_SUPERVISED:-0} == 1 ]] || exit 2
cooling_output=$1
shift
[[ $cooling_output == build/current-vae-threads-*-cooling.json && ${1:-} == -- ]] || exit 2
shift
bash scripts/check_environment.sh
.venv/bin/python scripts/wait_thermal_ready.py --output "$cooling_output"
exec .venv/bin/python scripts/sample_ve_memory.py --timeout 600 -- "$@"
