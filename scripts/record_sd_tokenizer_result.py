"""Publish tokenizer reuse only after numerical, lifecycle and ABBA gates."""
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
    for name in ('benchmark', 'probe', 'six', 'four', 'benchmark-log', 'probe-log', 'six-log', 'four-log', 'build-log'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    def local(p):
        p = p.resolve()
        p.relative_to(ROOT/'build')
        return p
    manifest = json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('build changed')
    if manifest['backend'] != 've':
        raise RuntimeError('native VE build required')
    binary = digest(ROOT/'build/sd-baseline-ve/bin/sd')
    checker = digest(ROOT/'tests/check_sd_resident.py')
    def native(folder, steps, cases, tokenizer):
        folder = local(folder)
        run = json.loads((folder/'summary.json').read_text())
        if (not run['completed'] or run['returncode'] or run['binary_sha256'] != binary or
                run['checker_sha256'] != checker or digest(folder/'check_sd_resident.py') != checker or
                run['mode'] != 'resident' or run['tokenizer_mode'] != tokenizer or
                run['steps'] != steps or run['threads'] != 8 or not run['shared_threadpool'] or
                [r['reference_case'] for r in run['requests']] != cases):
            raise RuntimeError('matching resident numerical run required')
        for i, row in enumerate(run['requests']):
            if (len(row['checks']) != 3+2*steps or
                    not all(c['passed'] and math.isfinite(c['relative_l2']) for c in row['checks']) or
                    len(row['trace_sha256']) != 4+3*steps or
                    row['weight_load_stages'] != (['clip', 'unet', 'vae'] if i == 0 else [])):
                raise RuntimeError('complete CPU checks and weight lifecycle required')
            for name, expected in row['trace_sha256'].items():
                if Path(name).name != name or digest(folder/('request%d' % i)/name) != expected:
                    raise RuntimeError('trace changed')
            if digest(folder/('request%d' % i)/'image.png') != row['png_sha256']:
                raise RuntimeError('PNG changed')
            timings = [p for p in row['profile'] if p['stage'] == 'clip' and p['part'] == 'tokenizer_init']
            if len(timings) != 1 or not math.isfinite(timings[0]['seconds']) or timings[0]['seconds'] < 0:
                raise RuntimeError('tokenizer initialization timing required')
        return run
    probe = native(args.probe, 1, [0, 1, 0], 'resident')
    six = native(args.six, 1, list(range(6)), 'resident')
    four = native(args.four, 4, [0, 0], 'resident')
    def identical(a, b):
        return a['trace_sha256'] == b['trace_sha256'] and a['png_sha256'] == b['png_sha256']
    if not identical(probe['requests'][0], probe['requests'][2]) or not identical(four['requests'][0], four['requests'][1]):
        raise RuntimeError('repeated prompts differ')
    folder = local(args.benchmark)
    benchmark = json.loads((folder/'summary.json').read_text())
    benchsha = digest(ROOT/'scripts/benchmark_sd_resident.py')
    if (not benchmark['completed'] or benchmark['kind'] != 'sd_turbo_tokenizer_abba' or
            benchmark['binary_sha256'] != binary or benchmark['checker_sha256'] != checker or
            digest(folder/'benchmark_sd_resident.py') != benchmark['benchmark_sha256'] or
            [r['mode'] for r in benchmark['runs']] != ['request', 'resident', 'resident', 'request']):
        raise RuntimeError('complete matching tokenizer ABBA required')
    if benchmark['benchmark_sha256'] != benchsha:
        recovery = benchmark.get('aggregation_recovery', {})
        backup = folder/'summary-before-recovery.json'
        if (recovery.get('recovery_sha256') != digest(ROOT/'scripts/recover_sd_tokenizer_abba.py') or
                recovery.get('original_summary_sha256') != digest(backup)):
            raise RuntimeError('audited metric recovery provenance required')
        original = json.loads(backup.read_text())
        if original['completed'] or original['runs'] != benchmark['runs'] or original['benchmark_sha256'] != benchmark['benchmark_sha256']:
            raise RuntimeError('metric recovery original evidence differs')
    runs = []
    for arm in benchmark['runs']:
        run = native(ROOT/arm['artifacts'], 1, [0, 0], arm['mode'])
        memory = json.loads((local(ROOT/arm['memory_artifacts'])/'summary.json').read_text())
        if not memory['completed'] or memory['returncode'] or memory['final_used_kib'] != memory['baseline_used_kib']:
            raise RuntimeError('completed memory sampling and recovery required')
        if run['model_revision'] != six['model_revision'] or run['process_seconds'] != arm['process_seconds']:
            raise RuntimeError('model or timing mismatch')
        runs.append(run)
    for i in range(2):
        if not all(identical(runs[0]['requests'][i], r['requests'][i]) for r in runs):
            raise RuntimeError('ABBA outputs differ')
    metrics = {}
    for key, index in [('process', None), ('request0', 0), ('request1', 1)]:
        value = lambda r: r['process_seconds'] if index is None else r['requests'][index]['request_seconds']
        before = (value(runs[0])+value(runs[3]))/2
        after = (value(runs[1])+value(runs[2]))/2
        if not all(math.isfinite(v) and v > 0 for v in (before, after)):
            raise RuntimeError('finite positive timings required')
        metrics[key] = {'request_seconds': before, 'resident_seconds': after, 'latency_reduction_percent': (1-after/before)*100}
        actual = benchmark['metrics'][key]
        if any(abs(actual[k]-v) > 1e-9 for k, v in metrics[key].items()):
            raise RuntimeError('published metric does not match original timings')
    extra_memory = {}
    for name in ('probe_log', 'six_log', 'four_log'):
        matches = re.findall(r'VE memory samples: (build/results/[^\s]+)', local(getattr(args, name)).read_text())
        if len(matches) != 1:
            raise RuntimeError('memory observation required for every extra validation')
        memory = json.loads((local(ROOT/matches[0])/'summary.json').read_text())
        if not memory['completed'] or memory['returncode'] or memory['final_used_kib'] != memory['baseline_used_kib']:
            raise RuntimeError('extra validation memory recovery required')
        extra_memory[name] = memory
    if not all(r['model_revision'] == six['model_revision'] for r in (probe, four)):
        raise RuntimeError('extra validation model mismatch')
    faster = metrics['process']['latency_reduction_percent'] > 0 and metrics['request1']['latency_reduction_percent'] > 0
    report = {'status': 'tokenizer_reuse_verified' if faster else 'tokenizer_reuse_not_faster',
              'recommended_for_validated_workload': faster, 'default_enabled': False,
              'scope': 'official SD-Turbo FP32 512, single VE, serialized 1/4-step requests; ABBA two processes per arm, diagnostic outputs included, filesystem cache uncontrolled; no production or arbitrary-model claim',
              'build': manifest, 'model_revision': six['model_revision'], 'metrics': metrics,
              'abba': benchmark, 'probe': probe, 'six': six, 'four': four, 'extra_memory': extra_memory,
              'temperatures': {name: thermal(local(getattr(args, name))) for name in ('benchmark_log', 'probe_log', 'six_log', 'four_log', 'build_log')},
              'fan_observation': fan_detail(local(args.benchmark_log))}
    safe(report)
    target = ROOT/'docs/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-turbo-tokenizer-reuse.json', time.gmtime())
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
