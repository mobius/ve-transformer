"""Capture source and reused object provenance without host-specific paths."""
from pathlib import Path
import hashlib,json,sys
folder=Path(sys.argv[1]);variant=sys.argv[2]
assert variant in ('prefill','projection')
files=['src/qwen_nlc_hook.cpp','src/qwen_dense_cache.h','tests/check_qwen_attention.cpp','tests/check_qwen_nlc.cpp','scripts/check_qwen15_projection.sh','scripts/build_qwen15_prefill.sh','scripts/record_qwen_prefill.py','build/qwen15-session/main.o','build/qwen15-session/factory.o',str(folder/'hook.o'),str(folder/'qwen-infer-ve')]
for name in ('check-attention','check-projection'):
 if (folder/name).exists():files.append(str(folder/name))
previous=json.loads(Path('build/qwen15-session/manifest.json').read_text())
for name in ('build/qwen15-session/main.o','build/qwen15-session/factory.o'):
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==previous['sha256'][name]
manifest={'upstream_revision':previous['upstream_revision'],'compiler':previous['compiler'],'math_mode':'fp64_accumulation','attention':'NLC for multiple columns; original path for one column','projection_tile_rows':1024 if variant=='projection' else 128,'reused_main_source_sha256':previous['sha256']['src/qwen_infer.cpp'],'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
