"""Verify real GEMM dimensions and timings from guarded native SD requests."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'SD_NLC_GEMM stage=(clip|unet|vae) m=(\d+) n=(\d+) k=(\d+) lda=(\d+) ldb=(\d+) ldc=(\d+) trans_a=N trans_b=T seconds=([0-9.]+)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def local(path):
        path = path.resolve()
        path.relative_to(ROOT/'build')
        return path
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    folder = local(args.result)
    native = json.loads((folder/'summary.json').read_text())
    manifest = json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('build source or binary changed')
    if (manifest['backend'] != 've' or not native['completed'] or native['returncode'] or
            native['binary_sha256'] != digest(ROOT/'build/sd-baseline-ve/bin/sd') or
            native['checker_sha256'] != digest(ROOT/'tests/check_sd_resident.py') or
            native['checker_sha256'] != digest(folder/'check_sd_resident.py') or
            native['mode'] != 'resident' or native['tokenizer_mode'] != 'resident' or
            native['steps'] != 1 or native['threads'] != 8 or not native['shared_threadpool'] or
            [r['reference_case'] for r in native['requests']] != [0, 0]):
        raise RuntimeError('completed matching two-request native profile required')
    if (native.get('gelu_mode')!='ve' or native.get('im2col_mode')!='rows_256' or
            native.get('vae_blas_threads')!=4 or native.get('vae_threads')!=8 or
            native.get('nlc_threads')!='unified' or native.get('binary_scalar_mode')!='ve' or
            native.get('operator_profile_enabled')):
        raise RuntimeError('matching current optimized non-operator-profile configuration required')
    checks=check_requests(folder,native,native['binary_sha256'])
    rows_dispatch(folder,native,'rows_256')
    log = (folder/'native.log').read_text()
    segments = re.findall(r'SD_REQUEST_BEGIN index=(\d+) resident=1\n(.*?)SD_REQUEST_END index=(\d+) seconds=([0-9.]+)', log, re.S)
    if len(segments) != 2 or 'SD_TURBO_NATIVE_COMPLETE' not in log:
        raise RuntimeError('complete native lifecycle required')
    requests = []
    for i, (begin, body, end, elapsed) in enumerate(segments):
        row = native['requests'][i]
        if int(begin) != i or int(end) != i or float(elapsed) != row['request_seconds']:
            raise RuntimeError('request timing or order differs')
        if len(row['checks']) != 5 or not all(c['passed'] and math.isfinite(c['relative_l2']) for c in row['checks']):
            raise RuntimeError('five finite CPU checks required')
        if len(row['trace_sha256']) != 7:
            raise RuntimeError('seven complete traces required')
        for name, expected in row['trace_sha256'].items():
            if Path(name).name != name or digest(folder/('request%d' % i)/name) != expected:
                raise RuntimeError('trace changed')
        if digest(folder/('request%d' % i)/'image.png') != row['png_sha256']:
            raise RuntimeError('PNG changed')
        groups = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
        records = PATTERN.findall(body)
        if len(records) != body.count('SD_NLC_GEMM '):
            raise RuntimeError('unrecognized GEMM record')
        for stage, *values in records:
            m, n, k, lda, ldb, ldc = map(int, values[:6])
            seconds = float(values[6])
            if (not all(0 < v < 2**31 for v in (m, n, k, lda, ldb, ldc)) or
                    lda < k or ldb < k or ldc < n or not math.isfinite(seconds) or seconds < 0):
                raise RuntimeError('invalid CBLAS dimensions, strides or timing')
            group = groups[(stage, m, n, k, lda, ldb, ldc)]
            group['calls'] += 1
            group['seconds'] += seconds
        stages = {}
        for stage in ('clip', 'unet', 'vae'):
            calls = sum(v['calls'] for key, v in groups.items() if key[0] == stage)
            seconds = sum(v['seconds'] for key, v in groups.items() if key[0] == stage)
            backend = [p['seconds'] for p in row['profile'] if p['stage'] == stage and p['part'] == 'backend_BLAS']
            if not calls or len(backend) != 1 or seconds > backend[0]+calls*1e-6+.001:
                raise RuntimeError('complete GEMM timings bounded by actual BLAS stage required')
            stages[stage] = {'gemm_calls': calls, 'gemm_seconds': seconds, 'backend_blas_seconds': backend[0]}
        shapes = []
        for (stage, m, n, k, lda, ldb, ldc), value in sorted(groups.items(), key=lambda item: item[1]['seconds'], reverse=True):
            shapes.append(dict(stage=stage, m=m, n=n, k=k, lda=lda, ldb=ldb, ldc=ldc,
                               trans_a='N', trans_b='T', **value))
        requests.append({'request': i, 'request_seconds': row['request_seconds'], 'stages': stages, 'shapes_by_total_seconds': shapes})
    first, second = native['requests']
    if first['trace_sha256'] != second['trace_sha256'] or first['png_sha256'] != second['png_sha256']:
        raise RuntimeError('repeat arrays and PNG differ')
    if {(s['stage'], s['m'], s['n'], s['k'], s['lda'], s['ldb'], s['ldc'], s['calls']) for s in requests[0]['shapes_by_total_seconds']} != {(s['stage'], s['m'], s['n'], s['k'], s['lda'], s['ldb'], s['ldc'], s['calls']) for s in requests[1]['shapes_by_total_seconds']}:
        raise RuntimeError('cold and hot GEMM shape/call sets differ')
    matches = re.findall(r'VE memory samples: (build/results/[^\s]+)', local(args.log).read_text())
    if len(matches) != 1:
        raise RuntimeError('memory samples required')
    memory_folder=local(ROOT/matches[0])
    memory = json.loads((memory_folder/'summary.json').read_text())
    memory_rows=list(csv.DictReader((memory_folder/'memory.csv').open()))
    if not memory_rows or max(int(r['used_kib']) for r in memory_rows)!=memory['sampled_highest_used_kib'] or int(memory_rows[-1]['used_kib'])!=131072:
        raise RuntimeError('raw memory peak and recovery differ')
    if not memory['completed'] or memory['returncode'] or memory['baseline_used_kib'] != memory['final_used_kib']:
        raise RuntimeError('memory recovery required')
    guard=local(args.log).read_text()
    temp=local(ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)',guard)[-1])
    fans=local(ROOT/re.findall(r'Fan observation:.*log=([^\s]+)',guard)[-1])
    thermal_rows=list(csv.DictReader(temp.open()))
    if not thermal_rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in thermal_rows):
        raise RuntimeError('invalid thermal evidence or stop reached')
    files=[folder/'summary.json',folder/'native.log',folder/'check_sd_resident.py',memory_folder/'summary.json',memory_folder/'memory.csv',local(args.log),temp,fans,ROOT/'build/sd-baseline-ve/manifest.json']
    output = {'status': 'nlc_gemm_profile_verified', 'speedup_measured': False,
              'scope': 'one native VE, SD-Turbo FP32 512, two identical one-step serialized requests, resident weights/tokenizer and shared8 generic pool and four-thread VAE BLAS; per-call profiling overhead not measured; diagnostic only',
              'build': manifest, 'model_revision': native['model_revision'], 'checks_passed': checks,
              'configuration':{'im2col_mode':'rows_256','gelu_mode':'ve','generic_threads':8,'vae_blas_threads':4},
              'publisher_sha256':digest(Path(__file__)),
              'audit_dependency_sha256':{name:digest(ROOT/name) for name in ('scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')},
              'artifact_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},
              'arrays_and_png_byte_identical': True, 'artifacts': str(folder.relative_to(ROOT)),
              'raw_native_log_sha256': digest(folder/'native.log'), 'requests': requests, 'memory': memory,
              'temperatures': {'test': thermal(local(args.log))},
              'fan_observation': fan_detail(local(args.log))}
    safe(output)
    target=args.output.resolve()
    target.relative_to(ROOT/'docs/results')
    target.write_text(json.dumps(output, indent=2)+'\n')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
