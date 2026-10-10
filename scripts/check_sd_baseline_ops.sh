#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised "$@"
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $(git -C build/vendor/sd-turbo-baseline/ggml rev-parse HEAD) == 6fcbd60bc72ac3f7ad43f78c87e535f2e6206f58 ]]
[[ -z $(git -C build/vendor/sd-turbo-baseline status --porcelain) ]]
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/nc++ \
 -std=c++17 -O1 -fno-fast-math -DGGML_MAX_NAME=128 -Ibuild/vendor/sd-turbo-baseline/ggml/include \
 build/sd-baseline-overlay/check_sd_ops.cpp -o build/sd-baseline-ve/check-sd-ops \
 -Wl,--start-group build/sd-baseline-ve/ggml/src/libggml.a \
 build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a build/sd-baseline-ve/ggml/src/ggml-blas/libggml-blas.a \
 build/sd-baseline-ve/ggml/src/libggml-base.a -Wl,--end-group \
 -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_sequential -pthread -ldl -lm \
 -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib
ve_exec -N 1 build/sd-baseline-ve/check-sd-ops
