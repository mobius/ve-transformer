#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/nlc-gemm
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
flags=(-O1 -fno-fast-math -fno-associative-math -I/opt/nec/ve/nlc/3.1.0/include -L/opt/nec/ve/nlc/3.1.0/lib -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -Wl,-rpath-link,/opt/nec/ve/nfort/5.4.1/lib)
"${limited[@]}" /opt/nec/ve/bin/ncc "${flags[@]}" src/ve_nlc_gemm_probe.c -o build/nlc-gemm/seq -lcblas -lblas_sequential
"${limited[@]}" /opt/nec/ve/bin/ncc "${flags[@]}" -fopenmp src/ve_nlc_gemm_probe.c -o build/nlc-gemm/omp -lcblas -lblas_openmp
.venv/bin/python scripts/record_nlc_gemm_build.py
