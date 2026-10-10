#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/im2col-rows
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fdiag-vector=2 -c src/ve_sd_turbo_im2col_rows.c -o build/im2col-rows/kernel.o
"${limited[@]}" /opt/nec/ve/bin/ncc -O1 -fno-fast-math -fno-associative-math -mno-vector tests/check_sd_im2col_rows.c build/im2col-rows/kernel.o -o build/im2col-rows/check
ve_exec -N 1 build/im2col-rows/check
