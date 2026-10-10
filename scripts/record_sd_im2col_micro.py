"""Publish complete 8-thread ABBA im2col microbenchmarks, including regressions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-reference', action='store_true')
    args = parser.parse_args()
    text = args.log.read_text()
    if 'Temperature summary:' not in text or 'FAIL' in text or 'oracle mismatch' in text:
        raise RuntimeError('completed guarded run required')
    pattern = (r'IM2COL_BENCH ic=(\d+) arm=(\d+) mode=(\w+) repeat=(-?\d+) '
               r'threads=8 seconds=([\d.]+) checked_elements=(\d+) '
               r'input_unchanged=1 guards=1 PASS')
    rows = []
    seen = set()
    for ic, arm, mode, repeat, seconds, elements in re.findall(pattern, text):
        ic, arm, repeat, elements = map(int, (ic, arm, repeat, elements))
        if (ic, arm, repeat) in seen or ic not in (128, 256) or arm not in range(4) or repeat not in (-1, 0, 1, 2):
            raise RuntimeError('unexpected/duplicate measurement')
        seen.add((ic, arm, repeat))
        if mode != ('rows' if arm in (1, 2) else 'channels') or elements != ic*512*512*9:
            raise RuntimeError('workload/order mismatch')
        if not 0 < float(seconds) < 600:
            raise RuntimeError('invalid timing')
        rows.append(dict(ic=ic, arm=arm, mode=mode, repeat=repeat,
                         seconds=float(seconds), checked_elements=elements))
    expected = {(ic, arm, repeat) for ic in (128, 256) for arm in range(4) for repeat in (-1, 0, 1, 2)}
    if seen != expected:
        raise RuntimeError('incomplete benchmark')
    metrics = []
    for ic in (128, 256):
        base = [r['seconds'] for r in rows if r['ic']==ic and r['mode']=='channels' and r['repeat']>=0]
        candidate = [r['seconds'] for r in rows if r['ic']==ic and r['mode']=='rows' and r['repeat']>=0]
        b, c = statistics.mean(base), statistics.mean(candidate)
        metrics.append(dict(ic=ic, channels_mean_seconds=b, rows_mean_seconds=c,
                            latency_reduction_percent=100*(1-c/b),
                            channels_samples=base, rows_samples=candidate))
    thermal = re.search(r'Temperature summary: cpu peak=([\d.]+)C, ve0 peak=([\d.]+)C, ve1 peak=([\d.]+)C, ve2 peak=([\d.]+)C', text)
    fans = re.search(r'Fan observation: (\d+) channels, (\d+) observed changes', text)
    if not thermal or not fans:
        raise RuntimeError('missing temperature/fan observations')
    files = ['src/ve_sd_turbo_im2col.c', 'src/ve_sd_turbo_im2col_rows.c',
             'tests/benchmark_sd_im2col_rows.c', 'scripts/benchmark_sd_im2col_rows.sh',
             'build/im2col-rows/baseline.o', 'build/im2col-rows/kernel.o',
             'build/im2col-rows/benchmark']
    if args.parallel_reference:
        files = [name.replace('tests/benchmark_sd_im2col_rows.c', 'tests/benchmark_sd_im2col_rows_parallel_reference.c').replace('scripts/benchmark_sd_im2col_rows.sh', 'scripts/benchmark_sd_im2col_rows_parallel_reference.sh').replace('build/im2col-rows/benchmark', 'build/im2col-rows/benchmark-parallel-reference') for name in files]
    result = dict(scope='standalone 8-thread OpenMP kernel timing; reset/oracle excluded; no whole-model gain claim',
                  rows=rows, metrics=metrics, full_bitwise_checks=len(rows),
                  total_checked_elements=sum(r['checked_elements'] for r in rows),
                  cpu_peak_c=float(thermal[1]), ve_peak_c=max(map(float,thermal.groups()[1:])),
                  fan_channels=int(fans[1]), fan_changes=int(fans[2]),
                  sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
                  log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
                  publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
