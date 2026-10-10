#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 2 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
mkdir -p build/sd-softmax-sum-probe
.venv/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=Path('build/accepted/softmax-shapes-20261009T201945Z');index=json.loads((archive/'archive-sha256.json').read_text())
objects=list(Path('build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('ve_sd_turbo_softmax_sum.c.o'));assert len(objects)==1
assert sha(objects[0])==index['actual-softmax-sum-object.o']
shutil.copy2(objects[0],'build/sd-softmax-sum-probe/baseline.o')
assert sha(Path('build/sd-baseline-ve/manifest.json'))==index['build/sd-baseline-ve/manifest.json']
print('Actual model O1 sum object bound')
PY
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fno-associative-math -mno-vector -Dsd_ve_softmax_sum_f32=sd_ve_softmax_sum_o2_f32 -c src/ve_sd_turbo_softmax_sum.c -o build/sd-softmax-sum-probe/candidate.o
"${limited[@]}" /opt/nec/ve/bin/nc++ -std=c++17 -O1 -fno-fast-math -fno-associative-math tests/benchmark_sd_softmax_sum.cpp build/sd-softmax-sum-probe/baseline.o build/sd-softmax-sum-probe/candidate.o -o build/sd-softmax-sum-probe/probe
ve_exec -N 1 build/sd-softmax-sum-probe/probe
