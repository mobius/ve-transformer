"""Record input-reuse build inputs and reused-object provenance."""
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1])
previous=json.loads(Path('build/qwen15-session/manifest.json').read_text())
files=['src/qwen_nlc_hook.cpp','src/qwen_dense_cache.h','scripts/build_qwen15_input_reuse.sh','scripts/record_qwen_input_reuse.py','scripts/check_qwen15_input_reuse.sh','tests/check_qwen_nlc.cpp','build/qwen15-session/main.o','build/qwen15-session/factory.o',str(folder/'hook.o'),str(folder/'qwen-infer-ve'),str(folder/'check-projection')]
hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
for name in ('build/qwen15-session/main.o','build/qwen15-session/factory.o'):
 assert hashes[name]==previous['sha256'][name]
manifest={'upstream_revision':previous['upstream_revision'],'compiler':previous['compiler'],'projection_tile_rows':1024,'diagnostic_profile':False,'reuse_input':True,'input_payload_max_bytes_per_worker':67108864,'input_capacity_overhead':'vector capacity and allocator overhead excluded','math_mode':'fp64_accumulation','build_script':'scripts/build_qwen15_input_reuse.sh','sha256':hashes}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
