#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 2 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
bash scripts/check_environment.sh
mkdir -p build/sd-gemm-pack-actual
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
mapfile -t objects < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-blas/CMakeFiles/ggml-blas.dir | rg '/ve_sd_turbo_gemm_pack\.c\.o$')
[[ ${#objects[@]} == 1 ]]
cp "${objects[0]}" build/sd-gemm-pack-actual/pack.o
"${limited[@]}" /opt/nec/ve/bin/ncc -O1 -fno-fast-math -mno-vector -fopenmp tests/check_sd_gemm_pack.c build/sd-gemm-pack-actual/pack.o -o build/sd-gemm-pack-actual/check
pack_run_dir=build/results/$(.venv/bin/python -c 'import time;print(time.strftime("%Y%m%dT%H%M%SZ-sd-gemm-pack-actual",time.gmtime()))')
mkdir -p "$pack_run_dir"
cp src/ve_sd_turbo_gemm_pack.c tests/check_sd_gemm_pack.c scripts/check_sd_gemm_pack_actual.sh build/sd-gemm-pack-actual/pack.o build/sd-gemm-pack-actual/check "$pack_run_dir/"
printf 'Actual pack kernel artifacts: %s\n' "$pack_run_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib OMP_NUM_THREADS=4 VE_OMP_NUM_THREADS=4 timeout --signal=TERM --kill-after=2 120 ve_exec -N 1 build/sd-gemm-pack-actual/check
