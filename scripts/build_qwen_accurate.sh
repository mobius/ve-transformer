#!/usr/bin/env bash
# Build the explicit FP64 accumulation experiment without replacing existing binaries.
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
[[ -f build/qwen-ve-precision/manifest.json ]]
cmake -S framework -B build/qwen-cpu -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_CCACHE=OFF -DGGML_LLAMAFILE=OFF
taskset -c 0-23 cmake --build build/qwen-cpu --target qwen-infer-cpu-accurate check-qwen-accurate-cpu check-qwen-accurate-primitives-cpu -j2
mkdir -p build/qwen-ve-accurate
inc=(-I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu")
strict=(-fno-fast-math -fno-associative-math -fno-reciprocal-math -mvector-sqrt-instruction -mvector-floating-divide-instruction -fdiag-inline=0 -fdiag-vector=0)
flags=(-O3 -std=c++17 "${strict[@]}" -DQWEN_ACCUM_FP64)
taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -c src/qwen_infer.cpp -o build/qwen-ve-accurate/main.o
taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -I/opt/nec/ve/nlc/3.1.0/include -DQWEN_NLC \
    -c src/qwen_nlc_hook.cpp -o build/qwen-ve-accurate/hook.o
math=(build/qwen-ve-accurate/hook.o build/qwen-ve-precision/ve_quant_kernels.o \
      -Wl,--wrap=ggml_cpu_extra_compute_forward -Wl,--wrap=ggml_vec_dot_f32)
libs=(build/llama-ve/ggml/src/libggml.a build/qwen-ve-precision/libggml-cpu.a build/qwen-ve-precision/libggml-base.a)
taskset -c 0-23 nc++ -fopenmp -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib \
    build/qwen-ve-accurate/main.o build/nec_llama_compat.o "${math[@]}" build/llama-ve/src/libllama.a "${libs[@]}" \
    -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen-ve-accurate/qwen-infer-ve
taskset -c 0-23 nc++ "${flags[@]}" -fopenmp "${inc[@]}" tests/check_qwen_nlc.cpp "${math[@]}" "${libs[@]}" \
    -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -L/opt/nec/ve/nlc/3.1.0/lib \
    -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen-ve-accurate/check-math
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
taskset -c 0-23 build/qwen-cpu/check-qwen-accurate-cpu
ve_exec -N 1 build/qwen-ve-accurate/check-math
taskset -c 0-23 nc++ "${flags[@]}" -fopenmp "${inc[@]}" tests/check_qwen_accurate.cpp "${math[@]}" "${libs[@]}" \
    -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -L/opt/nec/ve/nlc/3.1.0/lib \
    -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen-ve-accurate/check-primitives
taskset -c 0-23 build/qwen-cpu/check-qwen-accurate-primitives-cpu
ve_exec -N 1 build/qwen-ve-accurate/check-primitives
.venv/bin/python scripts/record_qwen_build.py --output build/qwen-ve-accurate/build-manifest.json --ve-flags="${flags[*]}"
