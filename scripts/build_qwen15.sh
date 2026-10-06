#!/usr/bin/env bash
# Isolated qwen2 factory; preserve all accepted 35B executors and archives.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
mkdir -p build/qwen15-ve
inc=(-I"$source_dir/src" -Ibuild/llama-ve/src -I"$source_dir/include" -I"$source_dir/ggml/include")
taskset -c 0-23 nc++ -O0 -std=c++17 -DGGML_USE_CPU -fdiag-inline=0 -fdiag-vector=0 "${inc[@]}" -c src/qwen15_factory.cpp -o build/qwen15-ve/factory.o
factory=(build/qwen15-ve/factory.o -Wl,--wrap=_Z18llama_model_create8llm_archRK18llama_model_params -Wl,--wrap=_Z18llama_model_createR18llama_model_loaderRK18llama_model_params)
libs=(build/llama-ve/ggml/src/libggml.a build/llama-ve/ggml/src/libggml-cpu.a build/llama-ve/ggml/src/libggml-base.a)
taskset -c 0-23 nc++ -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib build/qwen_infer.o build/nec_llama_compat.o build/ve_quant_kernels.o build/ve_quant_wrap.o -Wl,--wrap=ggml_vec_dot_q4_K_q8_K -Wl,--wrap=ggml_vec_dot_q5_K_q8_K -Wl,--wrap=ggml_vec_dot_q6_K_q8_K -Wl,--wrap=ggml_vec_dot_q8_0_q8_0 "${factory[@]}" build/llama-ve/src/libllama.a "${libs[@]}" -lpthread -ldl -lm -o build/qwen15-ve/qwen-infer-quant
libs=(build/llama-ve/ggml/src/libggml.a build/qwen-ve-precision/libggml-cpu.a build/qwen-ve-precision/libggml-base.a)
taskset -c 0-23 nc++ -fopenmp -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib build/qwen-ve-accurate/main.o build/nec_llama_compat.o build/qwen-ve-accurate/hook.o build/qwen-ve-precision/ve_quant_kernels.o -Wl,--wrap=ggml_cpu_extra_compute_forward -Wl,--wrap=ggml_vec_dot_f32 "${factory[@]}" build/llama-ve/src/libllama.a "${libs[@]}" -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen15-ve/qwen-infer-accurate
