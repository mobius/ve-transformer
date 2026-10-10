#!/usr/bin/env bash
# Fixed independent image framework; all generated artifacts stay in build/.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised "$@"
fi
shift
backend=${1:-ve}
[[ $backend == cpu || $backend == ve ]] || exit 2
bash scripts/check_environment.sh
source_dir=build/vendor/stable-diffusion.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == a1ded76da5818803fca97a3b433669ef727d32cf ]]
[[ $(git -C "$source_dir/ggml" rev-parse HEAD) == 89c4413f5da6fb20cc796f16033d37f129be81fd ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
 .venv/bin/python scripts/prepare_sd_overlay.py
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
export SD_BUILD_DUTY_PERCENT=${SD_BUILD_DUTY_PERCENT:-75}
limited=(.venv/bin/python scripts/cpu_duty.py --percent "$SD_BUILD_DUTY_PERCENT" -- taskset -c 0-3)
options=(-DCMAKE_PROJECT_INCLUDE="$PWD/cmake/nec-sd-overrides.cmake" -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SHARED_LIBS=OFF
 -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_CCACHE=OFF -DGGML_BLAS=ON
 -DGGML_BLAS_VENDOR=Generic -DCMAKE_C_FLAGS_RELEASE='-O1 -DNDEBUG -fno-fast-math'
 -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG -fno-fast-math')
if [[ $backend == ve ]]; then
 options+=(-DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/nec-ve.cmake"
  -DBLAS_INCLUDE_DIRS=/opt/nec/ve/nlc/3.1.0/include
  '-DBLAS_LIBRARIES=/opt/nec/ve/nlc/3.1.0/lib/libcblas.so;/opt/nec/ve/nlc/3.1.0/lib/libblas_sequential.so'
  -DBLAS_FOUND=ON)
fi
"${limited[@]}" cmake -S "$source_dir" -B "build/sd-$backend" "${options[@]}"
"${limited[@]}" cmake --build "build/sd-$backend" --target sd-cli -j 1
.venv/bin/python scripts/record_sd_build.py --backend "$backend"
