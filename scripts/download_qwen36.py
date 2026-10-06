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

REPO = 'unsloth/Qwen3.6-35B-A3B-GGUF'
REVISION = 'a483e9e6cbd595906af30beda3187c2663a1118c'
FILE = 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
SIZE = 22134528992
SHA256 = 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61'


def main():
    target = ROOT / 'build/models/qwen36-35b-a3b'
    target.mkdir(parents=True, exist_ok=True)
    existing = target / FILE
    needed = 0 if existing.exists() else SIZE
    if shutil.disk_usage(target).free < needed + 2 * 1024**3:
        raise RuntimeError('insufficient project disk space')
    print('Downloading pinned public Qwen3.6-35B-A3B quantization; expected bytes:', SIZE, flush=True)
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
    manifest = dict(model='Qwen/Qwen3.6-35B-A3B', quantization_repository=REPO,
                    revision=REVISION, file=FILE, bytes=SIZE, sha256=SHA256,
                    note='Third-party UD-Q4_K_M; CPU and VE must use identical file. '
                         'Digest verifies artifact integrity, not quantization quality.')
    (target / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Size and SHA256 verified; model stored in project-local ignored build/models.', flush=True)


if __name__ == '__main__':
    main()
