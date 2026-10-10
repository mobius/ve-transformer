#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised "$@"
fi
shift
bash scripts/check_environment.sh
bash scripts/prepare_sd_baseline_source.sh
.venv/bin/python scripts/prepare_sd_weight_index.py
.venv/bin/python scripts/prepare_sd_baseline_overlay.py
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
export SD_BUILD_DUTY_PERCENT=${SD_BUILD_DUTY_PERCENT:-50}
nlc_mode=${SD_NLC_MODE:-sequential}
case "$nlc_mode" in
 sequential) nlc_openmp=OFF; nlc_library=libblas_sequential.so ;;
 openmp) nlc_openmp=ON; nlc_library=libblas_openmp.so ;;
 *) printf 'SD_NLC_MODE must be sequential or openmp\n' >&2; exit 2 ;;
esac
export SD_NLC_MODE="$nlc_mode"
export SD_GGML_OPENMP=${SD_GGML_OPENMP:-0}
case "$SD_GGML_OPENMP" in
 0) ggml_openmp=OFF ;;
 1) ggml_openmp=ON; [[ "$nlc_mode" == openmp ]] || { printf 'Unified runtime requires OpenMP NLC\n' >&2; exit 2; } ;;
 *) printf 'SD_GGML_OPENMP must be 0 or 1\n' >&2; exit 2 ;;
esac
limited=(.venv/bin/python scripts/cpu_duty.py --percent "$SD_BUILD_DUTY_PERCENT" -- taskset -c 0-3)
"${limited[@]}" cmake -S build/vendor/sd-turbo-baseline -B build/sd-baseline-ve \
 -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/nec-sd-baseline-toolchain.cmake" \
 -DCMAKE_PROJECT_INCLUDE="$PWD/cmake/nec-sd-baseline-overrides.cmake" \
 -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SHARED_LIBS=OFF -DSD_FLASH_ATTN=OFF \
 -DSD_NLC_OPENMP="$nlc_openmp" \
 -DSD_GGML_OPENMP="$ggml_openmp" \
 -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_CCACHE=OFF \
 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Generic -DBLAS_FOUND=ON \
 -DBLAS_INCLUDE_DIRS=/opt/nec/ve/nlc/3.1.0/include \
 "-DBLAS_LIBRARIES=/opt/nec/ve/nlc/3.1.0/lib/libcblas.so;/opt/nec/ve/nlc/3.1.0/lib/$nlc_library" \
 -DCMAKE_C_FLAGS_RELEASE='-O1 -DNDEBUG -fno-fast-math' \
 -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG -fno-fast-math'
"${limited[@]}" cmake --build build/sd-baseline-ve --target sd -j 1
.venv/bin/python scripts/record_sd_baseline_build.py
