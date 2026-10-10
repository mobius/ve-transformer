"""Publish verified CLIP tokenizer timing evidence without claiming a speedup."""
import argparse
import hashlib
import json
import math
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
    manifest = json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('build source or binary changed')
    if (not report['completed'] or report['returncode'] or report['mode'] != 'resident'
            or report['steps'] != 1 or not report['shared_threadpool']
            or report['binary_sha256'] != digest(ROOT/'build/sd-baseline-ve/bin/sd')
            or report['checker_sha256'] != digest(folder/'check_sd_resident.py')):
        raise RuntimeError('completed native resident timing run required')
    requests = report['requests']
    if [r['reference_case'] for r in requests] != [0, 1, 0]:
        raise RuntimeError('different prompt followed by repeated original required')
    timings = []
    for i, request in enumerate(requests):
        if len(request['checks']) != 5 or not all(c['passed'] and math.isfinite(c['relative_l2']) for c in request['checks']):
            raise RuntimeError('five finite CPU checks required per request')
        for name, expected in request['trace_sha256'].items():
            if digest(folder/('request%d' % i)/name) != expected:
                raise RuntimeError('trace changed')
        if digest(folder/('request%d' % i)/'image.png') != request['png_sha256']:
            raise RuntimeError('PNG changed')
        parts = {}
        for name in ('tokenizer_init', 'tokenizer_encode', 'tokenizer_tensor', 'graph_compute', 'stage_total'):
            rows = [p for p in request['profile'] if p['stage'] == 'clip' and p['part'] == name]
            if len(rows) != 1 or rows[0]['calls'] != 1 or not math.isfinite(rows[0]['seconds']) or rows[0]['seconds'] < 0:
                raise RuntimeError('unique finite CLIP timing required')
            parts[name] = rows[0]['seconds']
        timings.append({'request': i, 'reference_case': request['reference_case'], 'request_seconds': request['request_seconds'], 'clip_seconds': parts})
    if requests[0]['trace_sha256'] != requests[2]['trace_sha256'] or requests[0]['png_sha256'] != requests[2]['png_sha256']:
        raise RuntimeError('repeated original prompt differs')
    output = {'status': 'tokenizer_reuse_probe_verified' if report.get('tokenizer_mode') == 'resident' else 'tokenizer_profile_verified',
              'tokenizer_mode': report.get('tokenizer_mode', 'request'), 'speedup_measured': False,
              'recommended_for_validated_workload': False, 'default_enabled': False,
              'scope': 'single VE, official SD-Turbo FP32 512, one step, three serialized requests; diagnostic timing only',
              'model_revision': report['model_revision'], 'build': manifest,
              'artifacts': str(folder.relative_to(ROOT)), 'checks_passed': 15,
              'repeated_original_byte_identical': True, 'timings': timings,
              'temperatures': {'test': thermal(args.log), 'build': thermal(args.build_log)},
              'fan_observation': fan_detail(args.log)}
    safe(output)
    target = ROOT/'docs/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-turbo-tokenizer-profile.json', time.gmtime())
    target.write_text(json.dumps(output, indent=2)+'\n')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
