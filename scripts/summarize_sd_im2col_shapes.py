"""Summarize native im2col dimensions; diagnostic timings are not speedups."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('N', 'IC', 'IH', 'IW', 'KH', 'KW', 'OH', 'OW',
          's0', 's1', 'p0', 'p1', 'd0', 'd1', 'threads')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT / 'build')
    summary = json.loads((run / 'summary.json').read_text())
    if not summary.get('completed'):
        raise RuntimeError('completed CPU-reference run required')
    log = run / 'native.log'
    rows = []
    counts = Counter()
    for line in log.read_text().splitlines():
        if not line.startswith('SD_IM2COL_SHAPE '):
            continue
        pairs = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if set(pairs) != {'stage', 'eligible', *FIELDS}:
            raise RuntimeError('unexpected shape record fields')
        stage = pairs['stage']
        if stage not in ('clip', 'unet', 'vae'):
            raise RuntimeError('unknown component')
        values = tuple(int(pairs[name]) for name in FIELDS)
        if any(value <= 0 for value in values[:10] + values[12:]):
            raise RuntimeError('invalid dimensions/stride/dilation/thread count')
        if int(pairs['eligible']) != 1:
            raise RuntimeError('ineligible im2col record; account for fallback first')
        counts[(stage, values)] += 1
    if not counts:
        raise RuntimeError('no im2col shape records')
    for (stage, values), count in sorted(counts.items()):
        shape = dict(zip(FIELDS, values))
        n, ic, ih, iw, kh, kw, oh, ow = values[:8]
        rows.append(dict(stage=stage, shape=shape, calls=count,
                         input_bytes=4*n*ic*ih*iw,
                         output_bytes=4*n*oh*ow*ic*kh*kw))
    dispatch = Counter()
    for stage, optimized, fallback in re.findall(
            r'SD_IM2COL stage=(clip|unet|vae) optimized_nodes=(\d+) fallback_nodes=(\d+)',
            log.read_text()):
        if int(fallback):
            raise RuntimeError('unexpected fallback')
        dispatch[stage] += int(optimized)
    observed = Counter()
    for row in rows:
        observed[row['stage']] += row['calls']
    if observed != +dispatch:
        raise RuntimeError('shape counts differ from component dispatch counts')
    result = dict(scope='diagnostic shape counts across all requests; no measured speedup',
                  run=str(run.relative_to(ROOT)),
                  native_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                  summary_sha256=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
                  publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  total_calls=dict(observed), shapes=rows)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Verified im2col shapes:', len(rows), 'unique;', dict(observed))


if __name__ == '__main__':
    main()
