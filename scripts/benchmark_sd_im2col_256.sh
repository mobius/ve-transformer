#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/im2col-256
mapfile -t baseline < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_im2col\.c\.o$')
mapfile -t candidate < <(rg --files --hidden --no-ignore build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir | rg '/ve_sd_turbo_im2col_rows\.c\.o$')
[[ ${#baseline[@]} == 1 && ${#candidate[@]} == 1 ]]
.venv/bin/python - <<'PY'
from pathlib import Path
import json,hashlib
proof=json.loads(Path('docs/results/20261009T133351Z-sd-turbo-im2col-abba.json').read_text())
target=Path('build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir')
for name,digest in proof['model_kernel_object_sha256'].items():
 objects=list(target.rglob(name+'.o'))
 assert len(objects)==1 and hashlib.sha256(objects[0].read_bytes()).hexdigest()==digest
 assert proof['model_kernel_optimization'][name]=='-O2'
print('Verified actual O2 model objects')
PY
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc -O1 -fno-fast-math -mno-vector -fopenmp tests/benchmark_sd_im2col_256.c "${baseline[0]}" "${candidate[0]}" -o build/im2col-256/benchmark
timeout --signal=TERM --kill-after=2 600 ve_exec -N 1 build/im2col-256/benchmark
