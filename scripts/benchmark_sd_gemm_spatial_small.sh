#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
bash scripts/check_environment.sh
mkdir -p build/sd-gemm-spatial-small
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fno-associative-math -fopenmp -I/opt/nec/ve/nlc/3.1.0/include -L/opt/nec/ve/nlc/3.1.0/lib -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -Wl,-rpath-link,/opt/nec/ve/nfort/5.4.1/lib tests/benchmark_sd_gemm_spatial_small.c -o build/sd-gemm-spatial-small/probe -lcblas -lblas_openmp
gemm_run_dir=build/results/$(.venv/bin/python -c 'import time; print(time.strftime("%Y%m%dT%H%M%SZ-sd-gemm-spatial-small", time.gmtime()))')
mkdir -p "$gemm_run_dir"
cp tests/benchmark_sd_gemm_spatial_small.c scripts/benchmark_sd_gemm_spatial_small.sh build/sd-gemm-spatial-small/probe "$gemm_run_dir/"
printf 'Small tile artifacts: %s\n' "$gemm_run_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib OMP_NUM_THREADS=4 VE_OMP_NUM_THREADS=4 ve_exec -N 1 build/sd-gemm-spatial-small/probe
