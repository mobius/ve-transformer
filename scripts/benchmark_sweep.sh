#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    monitor_options=()
    if [ "${1:-}" = '--cpu-only' ]; then monitor_options=(--cpu-only); fi
    exec python3 scripts/temperature_guard.py "${monitor_options[@]}" -- \
        bash "$0" --temperature-supervised "$@"
fi
shift
cpu_only=0
if [ "${1:-}" = '--cpu-only' ]; then cpu_only=1; shift; fi
if [ "$cpu_only" -eq 1 ] && [ "$#" -ne 0 ]; then
    printf 'usage: bash scripts/benchmark_sweep.sh [--cpu-only | slot ...]\n' >&2
    exit 2
fi
nodes=("$@")
if [ "$cpu_only" -eq 0 ]; then
    bash scripts/check_environment.sh
    if [ "${#nodes[@]}" -eq 0 ]; then
        for slot in /dev/veslot[0-9]*; do
            [ -L "$slot" ] || continue
            nodes+=("${slot#/dev/veslot}")
        done
    fi
    for node in "${nodes[@]}"; do
        [[ "$node" =~ ^[0-9]+$ ]] || { printf 'Invalid slot ID\n' >&2; exit 2; }
    done
fi
make test build/benchmark-cpu
if [ "$cpu_only" -eq 0 ]; then make ve; fi
libs=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib
export VE_LD_LIBRARY_PATH="$libs${VE_LD_LIBRARY_PATH:+:$VE_LD_LIBRARY_PATH}"
mkdir -p build/results
stamp=$(date -u +%Y%m%dT%H%M%SZ)
shapes=('32 64 4 128 50' '128 256 8 1024 10' '256 512 8 2048 5')
for shape in "${shapes[@]}"; do
    read -r -a args <<< "$shape"
    label="t${args[0]}-d${args[1]}"
    printf 'CPU %s\n' "$label"
    timeout 60 ./build/benchmark-cpu "${args[@]}" \
        | tee "build/results/${stamp}-cpu-${label}.txt"
    for node in "${nodes[@]}"; do
        printf 'VE slot %s %s\n' "$node" "$label"
        timeout 60 /opt/nec/ve/bin/ve_exec -N "$node" ./build/benchmark-ve "${args[@]}" \
            | tee "build/results/${stamp}-ve${node}-${label}.txt"
    done
done
