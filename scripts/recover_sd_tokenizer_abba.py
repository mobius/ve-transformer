"""Recover metrics from four completed native runs after aggregation failed."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    folder.relative_to(ROOT/'build')
    path = folder/'summary.json'
    report = json.loads(path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if report['completed'] or report['kind'] != 'sd_turbo_tokenizer_abba' or [r['mode'] for r in report['runs']] != ['request', 'resident', 'resident', 'request']:
        raise RuntimeError('four completed arms with unfinished tokenizer aggregation required')
    if digest(folder/'benchmark_sd_resident.py') != report['benchmark_sha256']:
        raise RuntimeError('executed benchmark snapshot changed')
    if digest(ROOT/'build/sd-baseline-ve/bin/sd') != report['binary_sha256'] or digest(ROOT/'tests/check_sd_resident.py') != report['checker_sha256']:
        raise RuntimeError('tested binary/checker changed')
    for arm in report['runs']:
        native_folder = (ROOT/arm['artifacts']).resolve()
        native_folder.relative_to(ROOT/'build')
        run = json.loads((native_folder/'summary.json').read_text())
        memory_folder = (ROOT/arm['memory_artifacts']).resolve()
        memory_folder.relative_to(ROOT/'build')
        memory = json.loads((memory_folder/'summary.json').read_text())
        if (not run['completed'] or run['returncode'] or run['mode'] != 'resident' or
                run['tokenizer_mode'] != arm['mode'] or run['binary_sha256'] != report['binary_sha256'] or
                run['checker_sha256'] != report['checker_sha256'] or digest(native_folder/'check_sd_resident.py') != report['checker_sha256'] or
                run['requests'] != arm['requests'] or run['process_seconds'] != arm['process_seconds'] or
                not memory['completed'] or memory['returncode'] or memory != arm['memory'] or
                memory['final_used_kib'] != memory['baseline_used_kib']):
            raise RuntimeError('original verified evidence differs')
        for i, request in enumerate(run['requests']):
            if run['steps'] != 1 or len(run['requests']) != 2 or request['reference_case'] != 0 or len(request['checks']) != 5 or not all(c['passed'] and math.isfinite(c['relative_l2']) for c in request['checks']):
                raise RuntimeError('complete CPU checks required')
            for name, expected in request['trace_sha256'].items():
                if Path(name).name != name or digest(native_folder/('request%d' % i)/name) != expected:
                    raise RuntimeError('trace changed')
            if digest(native_folder/('request%d' % i)/'image.png') != request['png_sha256']:
                raise RuntimeError('PNG changed')
    runs = report['runs']
    for i in range(2):
        first = runs[0]['requests'][i]
        if not all(r['requests'][i]['trace_sha256'] == first['trace_sha256'] and r['requests'][i]['png_sha256'] == first['png_sha256'] for r in runs):
            raise RuntimeError('ABBA outputs differ')
    metrics = {}
    for key, index in [('process', None), ('request0', 0), ('request1', 1)]:
        value = lambda r: r['process_seconds'] if index is None else r['requests'][index]['request_seconds']
        before, after = (value(runs[0])+value(runs[3]))/2, (value(runs[1])+value(runs[2]))/2
        if not all(math.isfinite(v) and v > 0 for v in (before, after)):
            raise RuntimeError('invalid timings')
        metrics[key] = {'request_seconds': before, 'resident_seconds': after, 'latency_reduction_percent': (1-after/before)*100}
    backup = folder/'summary-before-recovery.json'
    if backup.exists():
        raise RuntimeError('recovery backup already exists')
    backup.write_bytes(path.read_bytes())
    report.update(completed=True, metrics=metrics, all_trace_and_png_bytes_identical=True,
                  decision='faster_pending_multi_prompt_validation' if metrics['process']['latency_reduction_percent'] > 0 else 'not_faster',
                  scope='two same-process requests, two runs per arm, diagnostics included, filesystem cache uncontrolled',
                  aggregation_recovery={'reason': 'tuple assignment omitted second mean; hardware runs completed before failure',
                                        'original_summary_sha256': digest(backup), 'recovery_sha256': digest(Path(__file__))})
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
