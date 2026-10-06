"""Check native interchange rejection and bridge restrictions."""
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
header = b'VTF32V1\0'+struct.pack('<5I', 1, 1, 1, 1, 0)
payload = struct.pack('<11f', 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1.25)
with tempfile.TemporaryDirectory(dir=root/'build') as tmp:
    source, target = Path(tmp)/'in.vtf', Path(tmp)/'out.f32'
    for content, expected in [
        (header+payload, 0), (b'bad', 2), (header+payload[:-4], 2),
        (header+payload+b'junk', 2),
        (header+payload[:-4]+struct.pack('<f', float('nan')), 2),
        (b'VTF32V1\0'+struct.pack('<5I', 0, 1, 1, 1, 0)+payload, 2),
        (b'VTF32V1\0'+struct.pack('<5I', 1, 1, 1, 1, 2)+payload, 2),
    ]:
        source.write_bytes(content)
        p = subprocess.run([str(root/'build/infer-cpu'), str(source), str(target)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p.returncode == expected, (expected, p.returncode, p.stderr)
        if expected == 0:
            assert struct.unpack('<f', target.read_bytes())[0] == 1.25
print('native interchange length/shape/nonfinite validation: PASS')
