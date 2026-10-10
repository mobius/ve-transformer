#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
build/venvs/sd-image/bin/python scripts/prepare_sd_cont_transpose_fixtures.py
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fno-associative-math -fdiag-vector=2 -c src/ve_sd_turbo_cont_transpose.c -o build/sd-cont-transpose-probe/candidate.o
"${limited[@]}" /opt/nec/ve/bin/nc++ -std=c++17 -O1 -fno-fast-math -fno-associative-math -fopenmp -DGGML_MAX_NAME=128 -Ibuild/vendor/sd-turbo-baseline/ggml/include \
 tests/benchmark_sd_cont_transpose.cpp build/sd-cont-transpose-probe/candidate.o -o build/sd-cont-transpose-probe/probe \
 -Wl,--start-group build/sd-baseline-ve/ggml/src/libggml.a build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a build/sd-baseline-ve/ggml/src/libggml-base.a -Wl,--end-group \
 -pthread -ldl -lm -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib
cont_run_dir=build/results/$(.venv/bin/python -c 'import time; print(time.strftime("%Y%m%dT%H%M%SZ-sd-cont-transpose-probe", time.gmtime()))')
mkdir -p "$cont_run_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib OMP_NUM_THREADS=8 VE_OMP_NUM_THREADS=8 \
 ve_exec -N 1 build/sd-cont-transpose-probe/probe "$PWD/build/sd-cont-transpose-probe/shapes.tsv" "$PWD/$cont_run_dir"
readelf -Ws build/sd-cont-transpose-probe/candidate.o > build/sd-cont-transpose-symbols.log
readelf -r build/sd-cont-transpose-probe/candidate.o > build/sd-cont-transpose-relocations.log
printf 'CONT transpose artifacts: %s\n' "$cont_run_dir"
