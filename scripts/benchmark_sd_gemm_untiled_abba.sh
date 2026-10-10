#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
.venv/bin/python scripts/benchmark_sd_gemm_untiled_abba.py --reference build/results/20261008T081800Z-sd-reference --proof docs/results/20261009T235254Z-sd-turbo-gemm-untiled-model.json
