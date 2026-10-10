#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mapfile -t objects < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_silu\.c\.o$')
[[ ${#objects[@]} == 1 && -f ${objects[0]} ]]
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O1 -fno-fast-math tests/check_sd_silu.c "${objects[0]}" -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -lm -o build/sd-baseline-ve/check-sd-silu
ve_exec -N 1 build/sd-baseline-ve/check-sd-silu
