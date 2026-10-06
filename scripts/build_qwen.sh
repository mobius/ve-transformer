#!/usr/bin/env bash
# Build only project-local artifacts from an exact upstream commit.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
    exec python3 scripts/temperature_guard.py --bmc-fans -- bash "$0" --temperature-supervised "$@"
fi
shift
backend=${1:-ve}
[[ $backend == cpu || $backend == ve ]] || { echo 'usage: bash scripts/build_qwen.sh [cpu|ve]' >&2; exit 2; }
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
revision=8d81559fa7b8bcac9f7c8b478858953486371f90
if [[ ! -d $source_dir/.git ]]; then
    mkdir -p build/vendor
    git clone --no-checkout --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$source_dir"
    git -C "$source_dir" checkout --detach "$revision"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || { echo 'unexpected llama.cpp revision' >&2; exit 1; }
[[ -z $(git -C "$source_dir" status --porcelain) ]] || { echo 'upstream checkout is modified' >&2; exit 1; }
if [[ $backend == cpu ]]; then
    cmake -S framework -B build/qwen-cpu -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_CCACHE=OFF -DGGML_LLAMAFILE=OFF
    taskset -c 0-23 cmake --build build/qwen-cpu --target qwen-infer qwen-quant-probe qwen-infer-cpu-float check-qwen-nlc-cpu -j 2
else
    .venv/bin/python scripts/prepare_llama_ve.py
    if [[ ! -f build/llama-ve/CMakeCache.txt ]]; then
    cmake -S "$source_dir" -B build/llama-ve -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/nec-ve.cmake" \
        -DCMAKE_PROJECT_INCLUDE="$PWD/cmake/nec-ve-overrides.cmake" \
        -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_COMMON=OFF -DBUILD_SHARED_LIBS=OFF
    fi
    taskset -c 0-23 cmake --build build/llama-ve --target llama -j 2
    bash scripts/link_qwen_ve.sh --temperature-supervised build/llama-ve/src/libllama.a
fi
