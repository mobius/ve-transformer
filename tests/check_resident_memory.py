"""Host-only resident worker lifetime check using the installed Valgrind."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from gpt_neo_ve import load_model
from resident_ve import export_bundle
from ve_transformer import ROOT

torch.set_num_threads(1)
model, tokenizer = load_model()
width = model.config.hidden_size
with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
    bundle = Path(tmp)/'model.bundle'
    export_bundle(model, bundle, 16)
    payload = bytearray()
    for tokens in (9, 1, 6, 9):
        payload.extend(struct.pack('<2I', tokens, 1))
        payload.extend(torch.zeros(tokens, width).numpy().tobytes())
    payload.extend(struct.pack('<2I', 0, 0))
    p = subprocess.run(['valgrind', '--error-exitcode=99', '--leak-check=full',
                        str(ROOT/'build/resident-cpu'), str(bundle)],
                       input=bytes(payload), capture_output=True, timeout=30)
    assert p.returncode == 0, 'resident memory check failed'
    assert p.stdout[:8] == b'VTREADY1'
    offset = 16
    for _ in range(4):
        magic, status, tokens, compute = struct.unpack_from('<8sIId', p.stdout, offset)
        assert magic == b'VTRES01\0' and status == 0 and tokens == 1
        offset += 24+4*width
    assert len(p.stdout) == offset
    print(p.stderr.decode(), end='')
print('resident CPU mixed-length memory/lifetime: PASS')
