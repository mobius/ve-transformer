#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    exec python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
        bash "$0" --temperature-supervised "$@"
fi
shift
nodes=("$@")
if [ "${#nodes[@]}" -eq 0 ]; then nodes=(1 2 3); fi
for node in "${nodes[@]}"; do
    [[ "$node" =~ ^[0-9]+$ ]] || { printf 'Invalid slot ID\n' >&2; exit 2; }
done
bash scripts/check_environment.sh
make fast build/shape-cpu build/check-workspace-cpu build/check-workspace-ve build/check-workspace-ve-fast
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
./build/check-workspace-cpu
python3 tests/check_reference.py ./build/transformer-cpu
for node in "${nodes[@]}"; do
    for mode in ve ve-fast; do
        /opt/nec/ve/bin/ve_exec -N "$node" "./build/check-workspace-${mode}"
    done
    .venv/bin/python tests/check_shapes.py --node "$node" --fast
    .venv/bin/python tests/check_shapes.py --node "$node" --fast --workspace
    for shape in '32 64 4 128 100' '128 256 8 1024 50' '256 512 8 2048 20'; do
        read -r -a params <<< "$shape"
        if [ -x build/benchmark-ve-fast-before-workspace ]; then
            printf 'slot=%s mode=before-workspace\n' "$node"
            timeout 60 /opt/nec/ve/bin/ve_exec -N "$node" ./build/benchmark-ve-fast-before-workspace "${params[@]}"
        fi
        printf 'slot=%s mode=malloc\n' "$node"
        timeout 60 /opt/nec/ve/bin/ve_exec -N "$node" ./build/benchmark-ve-fast "${params[@]}"
        printf 'slot=%s mode=reuse\n' "$node"
        timeout 60 /opt/nec/ve/bin/ve_exec -N "$node" ./build/benchmark-ve-fast "${params[@]}" --workspace
    done
done
.venv/bin/python tests/check_pytorch.py --fast-only
.venv/bin/python examples/tiny_model.py --backend fast
/opt/nec/ve/bin/ve_exec -N "${nodes[0]}" ./build/benchmark-ve-fast 256 512 8 2048 30 --workspace --profile
