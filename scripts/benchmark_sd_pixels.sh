#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/sd-pixels-probe
build/venvs/sd-image/bin/python scripts/prepare_sd_pixels_fixtures.py
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
flags=(-DNDEBUG -fno-fast-math -fno-associative-math -std=c++11)
"${limited[@]}" /opt/nec/ve/bin/ncc -O0 -fno-inline "${flags[@]}" -c tests/sd_pixels_reference.cpp -o build/sd-pixels-probe/reference.o
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fdiag-vector=2 "${flags[@]}" -c src/ve_sd_turbo_pixels.cpp -o build/sd-pixels-probe/candidate.o
"${limited[@]}" /opt/nec/ve/bin/nc++ -O1 "${flags[@]}" tests/benchmark_sd_pixels.cpp build/sd-pixels-probe/reference.o build/sd-pixels-probe/candidate.o -o build/sd-pixels-probe/probe
pixel_run_dir=build/results/$(.venv/bin/python -c 'import time; print(time.strftime("%Y%m%dT%H%M%SZ-sd-pixels-probe", time.gmtime()))')
mkdir -p "$pixel_run_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib ve_exec -N 1 build/sd-pixels-probe/probe "$PWD/build/sd-pixels-probe/fixtures/fixtures.tsv" "$PWD/$pixel_run_dir"
printf 'Pixel artifacts: %s\n' "$pixel_run_dir"
