#!/usr/bin/env bash
# Isolated native MTP executable, preserving accepted inference binaries.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
output_dir=build/qwen36-mtp
mkdir -p "$output_dir/source"
inc=(-I"$source_dir/include" -I"$source_dir/src" -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu" -Isrc)
flags=(-O3 -std=c++17 -fno-fast-math -fno-associative-math -fno-reciprocal-math -mvector-sqrt-instruction -mvector-floating-divide-instruction -fdiag-inline=0 -fdiag-vector=0 -DQWEN_ACCUM_FP64 -DQWEN_TIMING)
export VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1
.venv/bin/python tests/check_cpu_duty.py
limited=(.venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3)
"${limited[@]}" nc++ "${flags[@]}" "${inc[@]}" -c src/qwen36_mtp.cpp -o "$output_dir/main.o"
"${limited[@]}" nc++ "${flags[@]}" "${inc[@]}" -c src/qwen36_factory.cpp -o "$output_dir/factory.o"
# Check the reused hook against the accepted input-reuse manifest.
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('build/qwen15-input-reuse/hook.o')
m=json.loads(Path('build/qwen15-input-reuse/manifest.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'][str(p)]
PY
libs=(build/llama-ve/ggml/src/libggml.a build/qwen-ve-precision/libggml-cpu.a build/qwen-ve-precision/libggml-base.a)
"${limited[@]}" nc++ -fopenmp -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib "$output_dir/main.o" build/nec_llama_compat.o \
 build/qwen15-input-reuse/hook.o build/qwen-ve-precision/ve_quant_kernels.o -Wl,--wrap=ggml_cpu_extra_compute_forward -Wl,--wrap=ggml_vec_dot_f32 \
 "$output_dir/factory.o" -Wl,--wrap=_Z18llama_model_create8llm_archRK18llama_model_params -Wl,--wrap=_Z18llama_model_createR18llama_model_loaderRK18llama_model_params \
 build/llama-ve/src/libllama.a "${libs[@]}" -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o "$output_dir/qwen-mtp-ve"
cp src/qwen36_mtp.cpp src/qwen36_factory.cpp scripts/build_qwen36_mtp.sh scripts/cpu_duty.py "$output_dir/source/"
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
files=['src/qwen36_mtp.cpp','src/qwen36_factory.cpp','scripts/build_qwen36_mtp.sh','scripts/cpu_duty.py','build/qwen36-mtp/main.o','build/qwen36-mtp/factory.o','build/qwen36-mtp/qwen-mtp-ve','build/qwen15-input-reuse/hook.o','build/llama-ve/src/libllama.a','build/qwen-ve-precision/ve_quant_kernels.o','build/qwen-ve-precision/libggml-cpu.a','build/qwen-ve-precision/libggml-base.a','build/llama-ve/ggml/src/libggml.a','build/nec_llama_compat.o']
out={'upstream_revision':'8d81559fa7b8bcac9f7c8b478858953486371f90','math_mode':'fp64_accumulation','hook':'accepted input-reuse hook','compiler_duty_percent':25,'sha256':{}}
for name in files:
 h=hashlib.sha256()
 with Path(name).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 out['sha256'][name]=h.hexdigest()
Path('build/qwen36-mtp/manifest.json').write_text(json.dumps(out,indent=2)+'\n')
PY
