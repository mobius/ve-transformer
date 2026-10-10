#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/sd-png-probe
build/venvs/sd-image/bin/python scripts/prepare_sd_png_fixtures.py
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
flags=(-DNDEBUG -fno-fast-math -fno-associative-math -Ibuild/vendor/sd-turbo-baseline/thirdparty)
"${limited[@]}" /opt/nec/ve/bin/ncc -O1 -std=c++11 "${flags[@]}" -c tests/sd_png_reference.cpp -o build/sd-png-probe/reference.o
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -std=c++11 -fdiag-vector=2 "${flags[@]}" -c src/ve_sd_turbo_png.cpp -o build/sd-png-probe/candidate.o
"${limited[@]}" /opt/nec/ve/bin/nc++ -O1 -std=c++11 "${flags[@]}" tests/benchmark_sd_png.cpp build/sd-png-probe/reference.o build/sd-png-probe/candidate.o -o build/sd-png-probe/probe
png_run_dir=build/results/$(.venv/bin/python -c 'import time; print(time.strftime("%Y%m%dT%H%M%SZ-sd-png-probe", time.gmtime()))')
mkdir -p "$png_run_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib ve_exec -N 1 build/sd-png-probe/probe "$PWD/build/sd-png-probe/fixtures/fixtures.tsv" "$PWD/$png_run_dir"
printf 'PNG artifacts: %s\n' "$png_run_dir"
