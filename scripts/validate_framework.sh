#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    exec python3 scripts/temperature_guard.py --post-seconds 5 -- \
        bash "$0" --temperature-supervised "$@"
fi
shift
if [ "$#" -ne 0 ]; then
    printf 'usage: bash scripts/validate_framework.sh\n' >&2
    exit 2
fi
bash scripts/check_environment.sh
make fast
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
.venv/bin/python tests/check_pytorch.py --fast-only
.venv/bin/python examples/tiny_model.py --backend fast
libs=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
export VE_LD_LIBRARY_PATH="$libs${VE_LD_LIBRARY_PATH:+:$VE_LD_LIBRARY_PATH}"
/opt/nec/ve/bin/ve_exec -N 1 ./build/benchmark-ve-fast 128 256 8 1024 30 --profile
/opt/nec/ve/bin/ve_exec -N 1 ./build/benchmark-ve-fast 256 512 8 2048 20 --profile
