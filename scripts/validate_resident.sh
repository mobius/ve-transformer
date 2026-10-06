#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    exec python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
        bash "$0" --temperature-supervised "$@"
fi
shift
node=${1:-1}
[[ "$node" =~ ^[0-9]+$ ]] || { printf 'Invalid slot ID\n' >&2; exit 2; }
if [ "$#" -gt 1 ]; then printf 'usage: bash scripts/validate_resident.sh [slot]\n' >&2; exit 2; fi
bash scripts/check_environment.sh
make fast resident build/infer-cpu build/check-workspace-cpu build/check-workspace-ve-fast
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
export HF_HOME="$root/build/hf-cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
./build/check-workspace-cpu
/opt/nec/ve/bin/ve_exec -N "$node" ./build/check-workspace-ve-fast
python3 tests/check_interchange.py
python3 tests/check_interchange_v2.py
.venv/bin/python tests/check_resident.py --node "$node" --repeats 3 --include-old
.venv/bin/python tests/benchmark_resident_last_head.py --node "$node"
.venv/bin/python tests/check_resident_odd_layers.py --node "$node"
