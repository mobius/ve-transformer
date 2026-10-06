#!/usr/bin/env bash
# Run in an environment with VE devices exposed; no privileged operations.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    exec python3 scripts/temperature_guard.py -- bash "$0" --temperature-supervised "$@"
fi
shift
bash scripts/check_environment.sh
nodes=("$@")
if [ "${#nodes[@]}" -eq 0 ]; then
    for slot in /dev/veslot[0-9]*; do
        [ -L "$slot" ] || continue
        nodes+=("${slot#/dev/veslot}")
    done
fi
if [ "${#nodes[@]}" -eq 0 ]; then
    printf 'No runtime slots found\n' >&2
    exit 1
fi
for node in "${nodes[@]}"; do
    [[ "$node" =~ ^[0-9]+$ ]] || { printf 'Invalid slot ID\n' >&2; exit 2; }
done
make test ve
# Scope runtime settings to this process and its children.
libs=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib
export VE_LD_LIBRARY_PATH="$libs${VE_LD_LIBRARY_PATH:+:$VE_LD_LIBRARY_PATH}"
mkdir -p build/results
stamp=$(date -u +%Y%m%dT%H%M%SZ)
for node in "${nodes[@]}"; do
    printf 'Validating runtime slot %s\n' "$node"
    # Sequential, small jobs: correctness before performance, no card saturation.
    timeout 30 python3 tests/check_reference.py /opt/nec/ve/bin/ve_exec -N "$node" \
        ./build/transformer-ve | tee "build/results/${stamp}-ve${node}-correctness.txt"
    timeout 30 /opt/nec/ve/bin/ve_exec -N "$node" ./build/benchmark-ve \
        | tee "build/results/${stamp}-ve${node}-benchmark.txt"
done
