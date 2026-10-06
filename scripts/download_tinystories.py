"""Download only the fixed author checkpoint into ignored project storage."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT/'build/hf-cache')
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download

REPO = 'roneneldan/TinyStories-1M'
REVISION = '77f1b168e219585646439073245fe87e56b3023e'
FILES = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'special_tokens_map.json', 'README.md', 'readme.md']

if __name__ == '__main__':
    target = ROOT/'build/models/tinystories-1m'
    snapshot_download(REPO, revision=REVISION, local_dir=target,
                      cache_dir=ROOT/'build/hf-cache', allow_patterns=FILES,
                      token=False, max_workers=2)
    hashes = {}
    for name in FILES:
        p = target/name
        if not p.is_file():
            raise RuntimeError('missing expected model file: '+name)
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(model=REPO, revision=REVISION, sha256=hashes,
                    license_metadata='not declared in author model card')
    (target/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('model={} revision={} downloaded_bytes={}'.format(
        REPO, REVISION, sum((target/name).stat().st_size for name in FILES)))
