"""Audit current selective-thread operator timings; no speedup inference."""
import argparse
import csv
import json
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--guard-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    native, memory = args.native.resolve(), args.memory.resolve()
    for folder in (native, memory):
        folder.relative_to(ROOT/'build')
    value = json.loads((native/'summary.json').read_text())
    manifest_path = ROOT/'build/sd-baseline-ve/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('build source or executable changed')
    if (value['vae_threads'] != 8 or value['vae_blas_threads'] != 4 or
            value['binary_scalar_mode'] != 've' or value['nlc_threads'] != 'unified' or
            value['mode'] != 'resident' or value['tokenizer_mode'] != 'resident' or
            not value['operator_profile_enabled'] or value['binary_shape_profile_enabled'] or
            value['steps'] != 1 or len(value['requests']) != 2 or
            value['binary_sha256'] != manifest['sha256']['build/sd-baseline-ve/bin/sd']):
        raise RuntimeError('matching selective-thread one-step profile required')
    if value.get('gelu_mode') != 've' or value.get('im2col_mode') != 'rows_256':
        raise RuntimeError('current optimized diagnostic configuration required')
    checks = check_requests(native, value, value['binary_sha256'])
    rows_dispatch(native, value, 'rows_256')
    raw = (native/'native.log').read_text()
    segments = re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+', raw, re.S)
    for request, body in zip(value['requests'], segments):
        operators = [{'stage': a, 'op': b, 'seconds': float(c), 'nodes': int(d)}
                     for a,b,c,d in re.findall(r'SD_OP_PROFILE stage=(clip|unet|vae) op=([A-Z_0-9]+) seconds=([0-9.]+) nodes=(\d+)', body)]
        if operators != request['operator_profile']:
            raise RuntimeError('operator timings differ from raw log')
    sampled = json.loads((memory/'summary.json').read_text())
    rows = list(csv.DictReader((memory/'memory.csv').open()))
    if (not sampled['completed'] or sampled['returncode'] or sampled['final_used_kib'] != 131072 or
            max(int(r['used_kib']) for r in rows) != sampled['sampled_highest_used_kib'] or
            int(rows[-1]['used_kib']) != 131072):
        raise RuntimeError('completed memory evidence and recovery required')
    hot = value['requests'][1]
    stages = {}
    for stage in ('clip','unet','vae'):
        operators = [dict(p) for p in hot['operator_profile'] if p['stage']==stage]
        total = sum(p['seconds'] for p in operators)
        cpu = sum(p['seconds'] for p in hot['profile'] if p['stage']==stage and p['part']=='backend_CPU')
        for p in operators:
            p['percent_of_operator_sum'] = p['seconds']/total*100
            p['percent_of_cpu_stage'] = p['seconds']/cpu*100
        stages[stage] = {'operator_sum_seconds':total, 'backend_cpu_seconds':cpu,
                         'operators':sorted(operators, key=lambda p:p['seconds'], reverse=True)}
    guard = args.guard_log.read_text()
    temperature = ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)', guard)[-1]
    fans = ROOT/re.findall(r'Fan observation:.*log=([^\s]+)', guard)[-1]
    thermal_rows = list(csv.DictReader(temperature.open()))
    if not thermal_rows or any(float(r['temperature_c'])>=float(r['stop_c']) for r in thermal_rows):
        raise RuntimeError('thermal stop threshold reached')
    files = [native/'summary.json',native/'native.log',native/'check_sd_resident.py',
             memory/'summary.json',memory/'memory.csv',args.guard_log,temperature,fans]
    report = {'status':'profile_audited','binary_sha256':value['binary_sha256'],
              'checker_sha256':value['checker_sha256'],'publisher_sha256':sha(Path(__file__)),
              'audit_helper_sha256':{name:sha(ROOT/name) for name in ('scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')},
              'im2col_mode':'rows_256','gelu_mode':'ve',
              'manifest_sha256':sha(manifest_path),'cpu_stage_checks':checks,
              'request_seconds':[r['request_seconds'] for r in value['requests']],
              'hot_stages':stages,'hot_component_profile':hot['profile'],
              'temperature':thermal(args.guard_log),'fan_detail':fan_detail(args.guard_log),
              'sampled_node_peak_gib':sampled['sampled_highest_used_kib']/1048576,
              'final_node_used_mib':128,
              'artifact_sha256':{str(p.resolve().relative_to(ROOT)):sha(p) for p in files},
              'scope':'one fixed one-step prompt, cold and hot request; includes operator timing and barriers; diagnostic only, no speedup claim'}
    safe(report)
    output = args.output.resolve()
    output.relative_to(ROOT/'docs/results')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Audited selective operator profile:',output.relative_to(ROOT))


if __name__ == '__main__':
    main()
