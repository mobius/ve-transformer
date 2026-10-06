"""Fetch a pinned public quantization into ignored project storage, verify SHA256."""
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / 'build/hf-cache')
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download

REPO = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
REVISION = '91cad51170dc346986eccefdc2dd33a9da36ead9'
FILE = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
SIZE = 1117320736
SHA256 = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'


def main():
    target = ROOT / 'build/models/qwen25-1.5b'
    target.mkdir(parents=True, exist_ok=True)
    existing = target / FILE
    needed = 0 if existing.exists() else SIZE
    if shutil.disk_usage(target).free < needed + 2 * 1024**3:
        raise RuntimeError('insufficient project disk space')
    print('Downloading pinned public Qwen2.5-1.5B-Instruct quantization; expected bytes:', SIZE, flush=True)
    path = Path(hf_hub_download(REPO, FILE, revision=REVISION,
                               local_dir=target, token=False))
    if path.stat().st_size != SIZE:
        raise RuntimeError('model size mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(32 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise RuntimeError('model SHA256 mismatch')
    manifest = dict(model='Qwen/Qwen2.5-1.5B-Instruct', quantization_repository=REPO,
                    revision=REVISION, file=FILE, bytes=SIZE, sha256=SHA256,
                    note='Official Q4_K_M; CPU and VE must use identical file. '
                         'Digest verifies artifact integrity, not quantization quality.')
    (target / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Size and SHA256 verified; model stored in project-local ignored build/models.', flush=True)


if __name__ == '__main__':
    main()
