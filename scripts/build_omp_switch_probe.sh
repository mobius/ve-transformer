#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
bash scripts/check_environment.sh
mkdir -p build/omp-switch
.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 /opt/nec/ve/bin/ncc \
 -O1 -fno-fast-math -fno-associative-math -fopenmp -I/opt/nec/ve/nlc/3.1.0/include \
 src/ve_omp_switch_probe.c -o build/omp-switch/probe -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lm \
 -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib -Wl,-rpath-link,/opt/nec/ve/nfort/5.4.1/lib
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,re,subprocess
files=[Path('src/ve_omp_switch_probe.c'),Path('scripts/build_omp_switch_probe.sh'),Path('build/omp-switch/probe')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
needed=re.findall(r'\(NEEDED\).*?\[([^]]+)\]',subprocess.check_output(['readelf','-d','build/omp-switch/probe'],text=True))
assert any(n.startswith('libblas_openmp') for n in needed) and not any(n.startswith('libblas_sequential') for n in needed)
report={'sha256':{str(p):sha(p) for p in files},'needed':needed,'library_sha256':{n:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/n) for n in ('libcblas.so','libblas_openmp.so')},'optimization':'O1 strict FP32, OpenMP, compiler duty25'}
Path('build/omp-switch/manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('OpenMP switch build captured')
PY
