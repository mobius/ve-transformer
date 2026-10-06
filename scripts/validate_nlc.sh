#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    exec python3 scripts/temperature_guard.py -- bash "$0" --temperature-supervised "$@"
fi
shift
bash scripts/check_environment.sh
if [ ! -x .venv/bin/python ]; then
    printf 'Create the local reference environment with bash scripts/setup_env.sh\n' >&2
    exit 1
fi
nodes=("$@")
if [ "${#nodes[@]}" -eq 0 ]; then
    for slot in /dev/veslot[0-9]*; do
        [ -L "$slot" ] || continue
        nodes+=("${slot#/dev/veslot}")
    done
fi
for node in "${nodes[@]}"; do
    [[ "$node" =~ ^[0-9]+$ ]] || { printf 'Invalid slot ID\n' >&2; exit 2; }
done
make test ve nlc build/shape-cpu build/shape-ve
libs=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
export VE_LD_LIBRARY_PATH="$libs${VE_LD_LIBRARY_PATH:+:$VE_LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
mkdir -p build/results
stamp=$(date -u +%Y%m%dT%H%M%SZ)
shapes=('32 64 4 128 50' '128 256 8 1024 10' '256 512 8 2048 5')
for node in "${nodes[@]}"; do
    python3 tests/check_reference.py /opt/nec/ve/bin/ve_exec -N "$node" \
        ./build/transformer-ve-nlc | tee "build/results/${stamp}-ve${node}-nlc-reference.txt"
    for mode in base nlc; do
        options=()
        if [ "$mode" = nlc ]; then options=(--nlc); fi
        .venv/bin/python tests/check_shapes.py --node "$node" "${options[@]}" \
            | tee "build/results/${stamp}-ve${node}-${mode}-shapes.txt"
    done
    for shape in "${shapes[@]}"; do
        read -r -a params <<< "$shape"
        for mode in base nlc; do
            binary=./build/benchmark-ve
            if [ "$mode" = nlc ]; then binary=./build/benchmark-ve-nlc; fi
            printf 'slot=%s backend=%s threads=%s\n' "$node" "$mode" "$OMP_NUM_THREADS"
            timeout 60 /opt/nec/ve/bin/ve_exec -N "$node" "$binary" "${params[@]}" \
                | tee "build/results/${stamp}-ve${node}-${mode}-t${params[0]}-d${params[1]}.txt"
        done
    done
done
