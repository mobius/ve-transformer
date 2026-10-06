#!/usr/bin/env bash
# Link full-model VE executors from a compiled library; always temperature guarded.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
    exec python3 scripts/temperature_guard.py --bmc-fans -- bash "$0" --temperature-supervised "$@"
fi
shift
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
    inc=(-I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/ggml/src")
    libs=("${1:-build/llama-ve/src/libllama.a}" build/llama-ve/ggml/src/libggml.a \
          build/llama-ve/ggml/src/libggml-cpu.a build/llama-ve/ggml/src/libggml-base.a)
    flags=(-O3 -std=c++17 -fdiag-inline=0 -fdiag-vector=0 -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib)
    taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -I"$source_dir/src" -c src/nec_llama_compat.cpp -o build/nec_llama_compat.o
    taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -c src/qwen_infer.cpp -o build/qwen_infer.o
    taskset -c 0-23 nc++ "${flags[@]}" build/qwen_infer.o build/nec_llama_compat.o "${libs[@]}" -lpthread -ldl -lm -o build/qwen-infer-ve-baseline
    taskset -c 0-23 ncc -O3 -std=c11 -fdiag-inline=0 -fdiag-vector=0 "${inc[@]}" -c src/ve_quant_kernels.c -o build/ve_quant_kernels.o
    taskset -c 0-23 ncc -O3 -std=c11 "${inc[@]}" -c src/ve_quant_wrap.c -o build/ve_quant_wrap.o
    wrap=(-Wl,--wrap=ggml_vec_dot_q4_K_q8_K -Wl,--wrap=ggml_vec_dot_q5_K_q8_K -Wl,--wrap=ggml_vec_dot_q6_K_q8_K -Wl,--wrap=ggml_vec_dot_q8_0_q8_0)
    taskset -c 0-23 nc++ "${flags[@]}" build/qwen_infer.o build/nec_llama_compat.o build/ve_quant_kernels.o build/ve_quant_wrap.o \
        "${wrap[@]}" "${libs[@]}" -lpthread -ldl -lm -o build/qwen-infer-ve
    taskset -c 0-23 nc++ "${flags[@]}" -DQWEN_NLC "${inc[@]}" -I"$source_dir/ggml/src/ggml-cpu" -I/opt/nec/ve/nlc/3.1.0/include \
        -c src/qwen_nlc_hook.cpp -o build/qwen_nlc_hook.o
    taskset -c 0-23 nc++ "${flags[@]}" -fopenmp build/qwen_infer.o build/nec_llama_compat.o build/ve_quant_kernels.o build/qwen_nlc_hook.o \
        -Wl,--wrap=ggml_cpu_extra_compute_forward "${libs[@]}" -L/opt/nec/ve/nlc/3.1.0/lib \
        -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen-infer-ve-nlc
    taskset -c 0-23 nc++ "${flags[@]}" -fopenmp "${inc[@]}" tests/qwen_quant_probe.cpp "${libs[@]:1}" -lpthread -ldl -lm -o build/qwen-quant-probe-ve
    taskset -c 0-23 nc++ "${flags[@]}" -fopenmp -DVE_QUANT_CANDIDATE "${inc[@]}" tests/qwen_quant_probe.cpp build/ve_quant_kernels.o \
        "${libs[@]:1}" -lpthread -ldl -lm -o build/qwen-quant-probe-ve-candidate
