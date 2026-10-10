#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/sd-gelu-probe
gelu_object=build/sd-gelu-probe/gelu.o
if [[ ${SD_GELU_USE_MODEL_OBJECT:-0} == 1 ]]; then
 mapfile -t objects < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_gelu\.c\.o$')
 [[ ${#objects[@]} == 1 && -f ${objects[0]} ]]
 gelu_object=${objects[0]}
else
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fno-associative-math -fno-reciprocal-math -mvector-intrinsic-check -fdiag-vector=2 -c src/ve_sd_turbo_gelu.c -o build/sd-gelu-probe/gelu.o
fi
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O1 -fno-fast-math -mno-vector tests/check_sd_gelu.c "$gelu_object" -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -lm -o build/sd-gelu-probe/check-gelu
timeout 30 ve_exec -N 1 build/sd-gelu-probe/check-gelu
