#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --temperature-supervised ]]; then
 exec .venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash "$0" --temperature-supervised
fi
bash scripts/check_environment.sh
source_dir=build/vendor/llama.cpp
[[ $(git -C "$source_dir" rev-parse HEAD) == 8d81559fa7b8bcac9f7c8b478858953486371f90 ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
folder=Path('build/models/qwen36-mtp-source')
p=json.loads((folder/'source-provenance.json').read_text())
assert p['completed'] and p['revision']=='995ad96eacd98c81ed38be0c5b274b04031597b0' and len(p['tensors'])==22
digest=hashlib.sha256()
with (folder/'model.safetensors').open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
p['assembled_sha256']=digest.hexdigest()
(folder/'source-provenance.json').write_text(json.dumps(p,indent=2)+'\n')
PY
mkdir -p build/models/qwen36-mtp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c 0-3 .venv/bin/python "$source_dir/convert_hf_to_gguf.py" build/models/qwen36-mtp-source --mtp --outtype f32 --outfile build/models/qwen36-mtp/mtp-f32.gguf
taskset -c 0-3 .venv/bin/python scripts/prepare_qwen36_mtp_gguf.py
