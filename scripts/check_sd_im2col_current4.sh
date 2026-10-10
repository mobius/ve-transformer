#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
bash scripts/check_environment.sh
mkdir -p build/im2col-current4
mapfile -t baseline < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_im2col\.c\.o$')
mapfile -t candidate < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_im2col_rows\.c\.o$')
[[ ${#baseline[@]} == 1 && ${#candidate[@]} == 1 ]]
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O1 -fno-fast-math -mno-vector -fopenmp tests/check_sd_im2col_current4.c "${baseline[0]}" "${candidate[0]}" -o build/im2col-current4/check
current4_dir=build/results/$(.venv/bin/python -c 'import time;print(time.strftime("%Y%m%dT%H%M%SZ-im2col-current4",time.gmtime()))')
mkdir -p "$current4_dir"
cp tests/check_sd_im2col_current4.c scripts/check_sd_im2col_current4.sh build/im2col-current4/check "${baseline[0]}" "${candidate[0]}" build/sd-baseline-ve/manifest.json "$current4_dir/"
printf 'Current four-thread row artifacts: %s\n' "$current4_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib OMP_NUM_THREADS=4 VE_OMP_NUM_THREADS=4 timeout --signal=TERM --kill-after=2 600 ve_exec -N 1 build/im2col-current4/check
