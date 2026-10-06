#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [ "${1:-}" != '--temperature-supervised' ]; then
    exec python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
        bash "$0" --temperature-supervised "$@"
fi
shift
bash scripts/check_environment.sh
make fast cpu build/infer-cpu build/check-workspace-cpu build/check-workspace-ve-fast
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
export HF_HOME="$root/build/hf-cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
python3 tests/check_interchange.py
python3 tests/check_interchange_v2.py
./build/check-workspace-cpu
.venv/bin/python tests/check_bridge_limits.py
.venv/bin/python tests/check_gpt_neo.py --nodes "$@"
