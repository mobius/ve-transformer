#!/usr/bin/env bash
# Rebuild and check the CPU/VE float activation path against independent math.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
    exec python3 scripts/temperature_guard.py --bmc-fans -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
taskset -c 0-23 cmake --build build/qwen-cpu --target check-qwen-nlc-cpu -j2
taskset -c 0-23 nc++ -O3 -std=c++17 -fopenmp -fdiag-inline=0 -fdiag-vector=0 \
    -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu" \
    tests/check_qwen_nlc.cpp build/qwen_nlc_hook.o build/ve_quant_kernels.o \
    -Wl,--wrap=ggml_cpu_extra_compute_forward \
    build/llama-ve/ggml/src/libggml.a build/llama-ve/ggml/src/libggml-cpu.a build/llama-ve/ggml/src/libggml-base.a \
    -L/opt/nec/ve/nlc/3.1.0/lib -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib \
    -lcblas -lblas_openmp -lpthread -ldl -lm -o build/check-qwen-nlc-ve
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
taskset -c 0-23 build/qwen-cpu/check-qwen-nlc-cpu
ve_exec -N 1 build/check-qwen-nlc-ve
