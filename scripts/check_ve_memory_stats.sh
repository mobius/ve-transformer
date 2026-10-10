#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O1 -fno-fast-math src/ve_memory_probe.c -o build/ve-memory-probe -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib
exec .venv/bin/python scripts/check_ve_memory_stats.py "$@"
