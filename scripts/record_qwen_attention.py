"""Record relative-path build identities; omit private build logs and host metadata."""
from pathlib import Path
import hashlib,json
files=['src/qwen_infer.cpp','src/qwen_nlc_hook.cpp','src/qwen_dense_cache.h','src/qwen15_factory.cpp','tests/check_qwen_attention.cpp','scripts/build_qwen15_attention.sh','build/qwen15-attention/main.o','build/qwen15-attention/hook.o','build/qwen15-attention/factory.o','build/qwen15-attention/qwen-infer-ve','build/qwen15-attention/check-attention']
manifest={'upstream_revision':'8d81559fa7b8bcac9f7c8b478858953486371f90','compiler':'NEC nc++ 5.4.1','precision':'FP64 accumulation after original F16 input rounding','features':['QWEN_ACCUM_FP64','QWEN_TIMING','QWEN_ATTENTION_NLC','QWEN_NLC'],'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
Path('build/qwen15-attention/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
