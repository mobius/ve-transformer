#!/usr/bin/env bash
# Resident Qwen2 executor; preserve previously accepted binaries and archives.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
mkdir -p build/qwen15-attention
taskset -c 0-23 nc++ -O0 -std=c++17 -DGGML_USE_CPU -fdiag-inline=0 -fdiag-vector=0 \
    -I"$source_dir/src" -Ibuild/llama-ve/src -I"$source_dir/include" -I"$source_dir/ggml/include" \
    -c src/qwen15_factory.cpp -o build/qwen15-attention/factory.o
inc=(-I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu")
strict=(-fno-fast-math -fno-associative-math -fno-reciprocal-math -mvector-sqrt-instruction -mvector-floating-divide-instruction -fdiag-inline=0 -fdiag-vector=0)
flags=(-O3 -std=c++17 "${strict[@]}" -DQWEN_ACCUM_FP64 -DQWEN_TIMING -DQWEN_ATTENTION_NLC)
taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -c src/qwen_infer.cpp -o build/qwen15-attention/main.o
taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -I/opt/nec/ve/nlc/3.1.0/include -DQWEN_NLC \
    -c src/qwen_nlc_hook.cpp -o build/qwen15-attention/hook.o
math=(build/qwen15-attention/hook.o build/qwen-ve-precision/ve_quant_kernels.o \
      -Wl,--wrap=ggml_cpu_extra_compute_forward -Wl,--wrap=ggml_vec_dot_f32)
libs=(build/llama-ve/ggml/src/libggml.a build/qwen-ve-precision/libggml-cpu.a build/qwen-ve-precision/libggml-base.a)
taskset -c 0-23 nc++ -fopenmp -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib \
    build/qwen15-attention/main.o build/nec_llama_compat.o "${math[@]}" build/qwen15-attention/factory.o -Wl,--wrap=_Z18llama_model_create8llm_archRK18llama_model_params -Wl,--wrap=_Z18llama_model_createR18llama_model_loaderRK18llama_model_params build/llama-ve/src/libllama.a "${libs[@]}" \
    -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen15-attention/qwen-infer-ve

taskset -c 0-23 nc++ "${flags[@]}" -fopenmp "${inc[@]}" tests/check_qwen_attention.cpp "${math[@]}" "${libs[@]}" -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen15-attention/check-attention
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
ve_exec -N 1 build/qwen15-attention/check-attention
.venv/bin/python scripts/record_qwen_attention.py
