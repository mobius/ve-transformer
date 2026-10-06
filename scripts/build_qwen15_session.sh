#!/usr/bin/env bash
# Resident Qwen2 executor; preserve previously accepted binaries and archives.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
mkdir -p build/qwen15-session
taskset -c 0-23 nc++ -O0 -std=c++17 -DGGML_USE_CPU -fdiag-inline=0 -fdiag-vector=0 \
    -I"$source_dir/src" -Ibuild/llama-ve/src -I"$source_dir/include" -I"$source_dir/ggml/include" \
    -c src/qwen15_factory.cpp -o build/qwen15-session/factory.o
inc=(-I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu")
strict=(-fno-fast-math -fno-associative-math -fno-reciprocal-math -mvector-sqrt-instruction -mvector-floating-divide-instruction -fdiag-inline=0 -fdiag-vector=0)
flags=(-O3 -std=c++17 "${strict[@]}" -DQWEN_ACCUM_FP64 -DQWEN_TIMING)
taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -c src/qwen_infer.cpp -o build/qwen15-session/main.o
taskset -c 0-23 nc++ "${flags[@]}" "${inc[@]}" -I/opt/nec/ve/nlc/3.1.0/include -DQWEN_NLC \
    -c src/qwen_nlc_hook.cpp -o build/qwen15-session/hook.o
math=(build/qwen15-session/hook.o build/qwen-ve-precision/ve_quant_kernels.o \
      -Wl,--wrap=ggml_cpu_extra_compute_forward -Wl,--wrap=ggml_vec_dot_f32)
libs=(build/llama-ve/ggml/src/libggml.a build/qwen-ve-precision/libggml-cpu.a build/qwen-ve-precision/libggml-base.a)
taskset -c 0-23 nc++ -fopenmp -Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib \
    build/qwen15-session/main.o build/nec_llama_compat.o "${math[@]}" build/qwen15-session/factory.o -Wl,--wrap=_Z18llama_model_create8llm_archRK18llama_model_params -Wl,--wrap=_Z18llama_model_createR18llama_model_loaderRK18llama_model_params build/llama-ve/src/libllama.a "${libs[@]}" \
    -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp -lpthread -ldl -lm -o build/qwen15-session/qwen-infer-ve

.venv/bin/python - <<'PYMANIFEST'
from pathlib import Path
import hashlib,json
files=['src/qwen_infer.cpp','src/qwen_nlc_hook.cpp','src/qwen_dense_cache.h','src/qwen15_factory.cpp','scripts/build_qwen15_session.sh','build/qwen15-session/main.o','build/qwen15-session/hook.o','build/qwen15-session/factory.o','build/qwen15-session/qwen-infer-ve']
report={'upstream_revision':'8d81559fa7b8bcac9f7c8b478858953486371f90','compiler':'NEC nc++ 5.4.1','math_mode':'fp64_accumulation','timing_macro':'QWEN_TIMING enabled in main and hook','sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
Path('build/qwen15-session/manifest.json').write_text(json.dumps(report,indent=2)+'\n')
PYMANIFEST
