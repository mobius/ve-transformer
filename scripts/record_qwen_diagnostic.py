"""Record isolated diagnostic build inputs and instrumentation mode."""
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1]);variant=sys.argv[2]
assert variant in ('control','profile','sampled')
previous=json.loads(Path('build/qwen15-session/manifest.json').read_text())
files=['src/qwen_nlc_hook.cpp','src/qwen_dense_cache.h','scripts/build_qwen15_diagnostic.sh','scripts/record_qwen_diagnostic.py','tests/check_qwen_nlc.cpp','build/qwen15-session/main.o','build/qwen15-session/factory.o',str(folder/'hook.o'),str(folder/'qwen-infer-ve'),str(folder/'check-projection')]
hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
for name in ('build/qwen15-session/main.o','build/qwen15-session/factory.o'):
 assert hashes[name]==previous['sha256'][name]
manifest={'upstream_revision':previous['upstream_revision'],'compiler':previous['compiler'],'projection_tile_rows':1024,'diagnostic_profile':variant!='control','sample_bits':6 if variant=='sampled' else 0,'math_mode':'fp64_accumulation','build_script':'scripts/build_qwen15_diagnostic.sh','sha256':hashes}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
