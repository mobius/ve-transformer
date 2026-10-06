#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
inc=(-I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu")
flags=(-O3 -std=c++17 -fno-fast-math -fno-associative-math -fno-reciprocal-math -fdiag-inline=0 -fdiag-vector=0 -DQWEN_ACCUM_FP64 -DQWEN_TIMING -DQWEN_PROJECTION_TILE_ROWS=1024 -DQWEN_PROJECTION_SMOKE)
taskset -c 0-23 nc++ "${flags[@]}" -fopenmp "${inc[@]}" tests/check_qwen_nlc.cpp build/qwen15-projection/hook.o build/qwen-ve-precision/ve_quant_kernels.o -Wl,--wrap=ggml_cpu_extra_compute_forward -Wl,--wrap=ggml_vec_dot_f32 build/llama-ve/ggml/src/libggml.a build/qwen-ve-precision/libggml-cpu.a build/qwen-ve-precision/libggml-base.a -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen15-projection/check-projection
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
ve_exec -N 1 build/qwen15-projection/check-projection
.venv/bin/python scripts/record_qwen_prefill.py build/qwen15-prefill prefill
.venv/bin/python scripts/record_qwen_prefill.py build/qwen15-projection projection
