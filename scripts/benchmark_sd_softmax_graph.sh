#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
shift
bash scripts/check_environment.sh
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
build/venvs/sd-image/bin/python scripts/prepare_sd_softmax_graph.py
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
"${limited[@]}" build/venvs/sd-image/bin/python - <<'PY'
import json,subprocess
from pathlib import Path
folder=Path('build/sd-softmax-graph-probe')
args=json.loads((folder/'compile-args.json').read_text())
subprocess.run(['/opt/nec/ve/bin/ncc']+args+['-c',str(folder/'sd-ggml-cpu.c'),'-o',str(folder/'sd-ggml-cpu.c.o')],check=True)
PY
"${limited[@]}" /opt/nec/ve/bin/ncc -O2 -fno-fast-math -fno-associative-math -fdiag-vector=2 -c src/ve_sd_turbo_softmax_scale.c -o build/sd-softmax-graph-probe/scale.o
/opt/nec/ve/bin/nar r build/sd-softmax-graph-probe/libggml-cpu.a build/sd-softmax-graph-probe/sd-ggml-cpu.c.o build/sd-softmax-graph-probe/scale.o
"${limited[@]}" /opt/nec/ve/bin/nc++ -std=c++17 -O1 -fno-fast-math -fno-associative-math -fopenmp -DGGML_MAX_NAME=128 -Ibuild/vendor/sd-turbo-baseline/ggml/include tests/benchmark_sd_softmax_graph.cpp -o build/sd-softmax-graph-probe/probe \
 -Wl,--start-group build/sd-baseline-ve/ggml/src/libggml.a build/sd-softmax-graph-probe/libggml-cpu.a build/sd-baseline-ve/ggml/src/libggml-base.a -Wl,--end-group -pthread -ldl -lm -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib
softmax_run_dir=build/results/$(.venv/bin/python -c 'import time; print(time.strftime("%Y%m%dT%H%M%SZ-sd-softmax-graph-probe", time.gmtime()))')
mkdir -p "$softmax_run_dir"
VE_LD_LIBRARY_PATH=/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib OMP_NUM_THREADS=8 VE_OMP_NUM_THREADS=8 ve_exec -N 1 build/sd-softmax-graph-probe/probe build/sd-softmax-graph-probe/shapes.tsv "$softmax_run_dir"
printf 'Softmax graph artifacts: %s\n' "$softmax_run_dir"
