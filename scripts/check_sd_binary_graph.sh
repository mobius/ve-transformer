#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
export VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib
export OMP_NUM_THREADS=8 VE_OMP_NUM_THREADS=8 SD_VE_BINARY_SCALAR=1
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/nc++ \
 -std=c++17 -O1 -fno-fast-math -fopenmp -DGGML_MAX_NAME=128 -Ibuild/vendor/sd-turbo-baseline/ggml/include \
 tests/check_sd_binary_graph.cpp -o build/sd-baseline-ve/check-sd-binary-graph \
 -Wl,--start-group build/sd-baseline-ve/ggml/src/libggml.a \
 build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a build/sd-baseline-ve/ggml/src/ggml-blas/libggml-blas.a \
 build/sd-baseline-ve/ggml/src/libggml-base.a -Wl,--end-group \
 -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -pthread -ldl -lm \
 -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib
ve_exec -N 1 build/sd-baseline-ve/check-sd-binary-graph |& tee build/sd-baseline-ve/binary-graph.log
.venv/bin/python - <<'PY'
from pathlib import Path
import re
text=Path('build/sd-baseline-ve/binary-graph.log').read_text()
rows=re.findall(r'SD_BINARY_DISPATCH stage=unknown op=(ADD|MUL) optimized=(\d+) fallback=(\d+) enabled=1',text)
expected=[]
for kind in range(4):
 for multiply in range(2):
  for inplace in range(2):
   for op in ('ADD','MUL'):
    active=op==('MUL' if multiply else 'ADD')
    optimized=kind==0 or (kind==3 and not inplace)
    expected.append((op,str(int(active and optimized)),str(int(active and not optimized))))
assert rows==expected,rows
assert 'BINARY_GRAPH_PASS cases=16' in text
print('BINARY_DISPATCH_PASS optimized=6 fallback=10 overlapping_b_rejected=2')
PY
