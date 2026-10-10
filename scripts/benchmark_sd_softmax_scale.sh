#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 2 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/sd-softmax-scale-probe
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
"${limited[@]}" /opt/nec/ve/bin/ncc -O1 -fno-fast-math -fno-associative-math -mno-vector -DSD_SCALE_BASELINE -Dsd_ve_softmax_scale_f32=sd_o1_softmax_scale_f32 -Dsd_ve_softmax_copy_scale_f32=sd_o1_softmax_copy_scale_f32 -c src/ve_sd_turbo_softmax_scale.c -o build/sd-softmax-scale-probe/baseline.o
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fno-associative-math -fdiag-vector=2 -c src/ve_sd_turbo_softmax_scale.c -o build/sd-softmax-scale-probe/candidate.o
"${limited[@]}" /opt/nec/ve/bin/nc++ -std=c++17 -O1 -fno-fast-math tests/benchmark_sd_softmax_scale.cpp build/sd-softmax-scale-probe/baseline.o build/sd-softmax-scale-probe/candidate.o -o build/sd-softmax-scale-probe/probe
folder=build/results/$(.venv/bin/python -c 'import time; print(time.strftime("%Y%m%dT%H%M%SZ-sd-softmax-scale-probe",time.gmtime()))')
mkdir -p "$folder"
ve_exec -N 1 build/sd-softmax-scale-probe/probe "$folder"
printf 'Softmax scale artifacts: %s\n' "$folder"
