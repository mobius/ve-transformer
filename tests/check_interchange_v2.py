"""Reject malformed extended decoder options and verify its tensor layout."""
from pathlib import Path
import math
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Matrices Q,K,V,O,up,down; two affine norms; six biases; input.
values = [0, 0, 0, 0.4, 0.2, 0.3, 1, 0, 1, 0,
          0.1, 0.2, 0.3, 0.5, 0.3, 0.25, 1.25]
payload = struct.pack('<17f', *values)

def header(activation=1, scale=1, window=1, epsilon=1e-5, causal=1):
    return b'VTF32V2\0'+struct.pack('<8If', 1, 1, 1, 1, causal,
                                   activation, scale, window, epsilon)

with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
    source, target = Path(tmp)/'input.vtf', Path(tmp)/'output.f32'
    cases = [(header()+payload, 0), (header()[:-1], 2),
             (header(activation=2)+payload, 2), (header(scale=2)+payload, 2),
             (header(window=1025)+payload, 2), (header(causal=0)+payload, 2),
             (header(epsilon=float('nan'))+payload, 2), (header(epsilon=0)+payload, 2),
             (header()+payload[:-4], 2), (header()+payload+b'junk', 2),
             (header()+payload[:-4]+struct.pack('<f', float('inf')), 2)]
    for content, expected in cases:
        source.write_bytes(content)
        result = subprocess.run([str(ROOT/'build/infer-cpu'), str(source), str(target)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        assert result.returncode == expected, (expected, result.returncode, result.stderr)
        if expected == 0:
            actual = struct.unpack('<f', target.read_bytes())[0]
            gelu = 0.5*0.3*(1+math.tanh(math.sqrt(2/math.pi)*(0.3+0.044715*0.3**3)))
            assert abs(actual-(1.25+0.4*0.3+0.5+0.3*gelu+0.25)) < 2e-6
print('v2 options/bias-layout/nonfinite/length rejection: PASS')
