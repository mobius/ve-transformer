"""Index verified official FP32 tensor headers without rewriting weight bytes."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'build/models/sd-turbo'
REVISION = 'b261bac6fd2cf515557d5d0707481eafa0485ec2'

def main():
    provenance = json.loads((MODEL / 'provenance.json').read_text())
    if not provenance['completed'] or provenance['revision'] != REVISION:
        raise RuntimeError('complete pinned official model required')
    for name, entry in provenance['files'].items():
        if not name.endswith('.safetensors'):
            continue
        path = MODEL / name
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(4 * 1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != entry['sha256'] or digest.hexdigest() != entry['official_lfs_sha256']:
            raise RuntimeError('official weight checksum mismatch')
        with path.open('rb') as source:
            header_size = struct.unpack('<Q', source.read(8))[0]
            if not 2 <= header_size <= 16 * 1024**2:
                raise RuntimeError('invalid header size')
            header = json.loads(source.read(header_size))
        rows = []
        spans = []
        for tensor, metadata in header.items():
            if tensor == '__metadata__':
                continue
            shape = metadata['shape']
            begin, end = metadata['data_offsets']
            if (metadata['dtype'] != 'F32' or not 1 <= len(shape) <= 4
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or type(begin) is not int or type(end) is not int
                    or begin < 0 or end - begin != 4 * math.prod(shape)
                    or end > path.stat().st_size - 8 - header_size
                    or not tensor or any(c.isspace() for c in tensor)):
                raise RuntimeError('unsupported or invalid official tensor')
            rows.append(' '.join([tensor, str(len(shape)), *map(str, shape), str(begin), str(end)]))
            spans.append((begin, end))
        spans.sort()
        if not spans or spans[0][0] != 0 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
            raise RuntimeError('tensor payload overlaps or gaps')
        if spans[-1][1] != path.stat().st_size - 8 - header_size:
            raise RuntimeError('tensor payload coverage mismatch')
        text = 'VE_SDTURBO_INDEX_V1 %d %d %d\n' % (path.stat().st_size, header_size, len(rows))
        text += '\n'.join(rows) + '\n'
        index = Path(str(path) + '.ve-index')
        if not index.exists() or index.read_text() != text:
            index.write_text(text)
        print('Verified FP32 index:', name, 'tensors', len(rows), flush=True)

if __name__ == '__main__':
    main()
