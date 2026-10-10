#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/im2col-rows
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
if [[ ${SD_IM2COL_USE_MODEL_OBJECT:-0} == 1 ]]; then
 mapfile -t objects < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_im2col_rows\.c\.o$')
 [[ ${#objects[@]} == 1 && -f ${objects[0]} ]]
 kernel_object=${objects[0]}
else
 "${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fdiag-vector=2 -c src/ve_sd_turbo_im2col_rows.c -o build/im2col-rows/kernel.o
 kernel_object=build/im2col-rows/kernel.o
fi
"${limited[@]}" /opt/nec/ve/bin/ncc -O1 -fno-fast-math -fno-associative-math -mno-vector -fopenmp tests/check_sd_im2col_rows_concurrent.c "$kernel_object" -o build/im2col-rows/check-concurrent
ve_exec -N 1 build/im2col-rows/check-concurrent
