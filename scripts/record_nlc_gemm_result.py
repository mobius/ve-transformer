"""Publish verified native GEMM probes without extrapolating full-image gains."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import time

from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--build-log', type=Path, required=True)
    args = parser.parse_args()
    folder = args.result.resolve()
    folder.relative_to(ROOT/'build')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = json.loads((folder/'summary.json').read_text())
    if (not report['completed'] or len(report['runs']) != 40 or
            report['benchmark_sha256'] != digest(ROOT/'scripts/benchmark_nlc_gemm.py') or
            report['benchmark_sha256'] != digest(folder/'benchmark_nlc_gemm.py') or
            report['shapes_sha256'] != digest(folder/'shapes.json') or
            report['source_profile_sha256'] != digest(ROOT/report['source_profile'])):
        raise RuntimeError('completed immutable native benchmark required')
    for name, expected in report['build']['sha256'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('probe build changed')
    for name, expected in report['build']['library_sha256'].items():
        if digest(Path('/opt/nec/ve/nlc/3.1.0/lib')/name) != expected:
            raise RuntimeError('NLC library changed')
    shapes = json.loads((folder/'shapes.json').read_text())['shapes']
    comparisons = []
    for i, shape in enumerate(shapes):
        rows = report['runs'][i*5:(i+1)*5]
        if [(r['mode'], r['requested_threads']) for r in rows] != [('seq', 1), ('omp', 1), ('omp', 2), ('omp', 4), ('omp', 8)]:
            raise RuntimeError('all five variants required in fixed order')
        for row in rows:
            if (row['shape_index'] != i or row['shape'] != shape or row['parallel_region_threads'] != row['requested_threads'] or
                    row['sample_checks_passed'] != 80 or not row['final_output_all_finite'] or
                    not math.isfinite(row['max_abs_sample_error']) or len(row['kernel_seconds']) != 3 or
                    not all(math.isfinite(t) and t > 0 for t in row['kernel_seconds'])):
                raise RuntimeError('verified numeric and timing evidence required')
            if abs(sum(row['kernel_seconds'])/3-row['mean_kernel_seconds']) > 1e-12:
                raise RuntimeError('kernel mean differs from measurements')
            path = folder/('case%d-%s%d.log' % (i, row['mode'], row['requested_threads']))
            if digest(path) != row['raw_log_sha256']:
                raise RuntimeError('raw native log changed')
        best = min(rows, key=lambda r: r['mean_kernel_seconds'])
        comparisons.append({'shape_index': i, 'shape': shape, 'sequential_seconds': rows[0]['mean_kernel_seconds'],
                            'best_tested_mode': best['mode'], 'best_tested_threads': best['requested_threads'],
                            'best_tested_seconds': best['mean_kernel_seconds'],
                            'kernel_speedup': rows[0]['mean_kernel_seconds']/best['mean_kernel_seconds']})
    matches = re.findall(r'VE memory samples: (build/results/[^\s]+)', args.log.read_text())
    if len(matches) != 1:
        raise RuntimeError('memory observation required')
    memory_folder = (ROOT/matches[0]).resolve()
    memory_folder.relative_to(ROOT/'build')
    memory = json.loads((memory_folder/'summary.json').read_text())
    if not memory['completed'] or memory['returncode'] or memory['baseline_used_kib'] != memory['final_used_kib']:
        raise RuntimeError('complete memory monitoring and recovery required')
    output = {'status': 'nlc_gemm_probe_verified', 'full_graph_speedup_measured': False,
              'recommended_for_full_graph': False, 'sample_checks_passed': 3200,
              'scope': report['scope']+'; synthetic matrices initialized serially; twenty sampled positions per call, final entire output checked finite; one process per variant and three measured calls, no ABBA',
              'comparisons': comparisons, 'benchmark': report, 'memory': memory, 'artifacts': str(folder.relative_to(ROOT)),
              'temperatures': {'build': thermal(args.build_log), 'test': thermal(args.log)},
              'fan_observation': fan_detail(args.log)}
    safe(output)
    target = ROOT/'docs/results'/time.strftime('%Y%m%dT%H%M%SZ-nlc-gemm-probe.json', time.gmtime())
    target.write_text(json.dumps(output, indent=2)+'\n')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
