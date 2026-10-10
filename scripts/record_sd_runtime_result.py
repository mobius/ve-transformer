"""Publish audited runtime ABBA, earlier candidate correctness and thermal data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from benchmark_sd_runtime import verify_snapshot
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--guard-log', type=Path, required=True)
    parser.add_argument('--build-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.benchmark.resolve()
    folder.relative_to(ROOT/'build')
    bench = json.loads((folder/'summary.json').read_text())
    if not bench['completed'] or bench['same_binary'] or [r['arm'] for r in bench['runs']] != ['baseline', 'candidate', 'candidate', 'baseline']:
        raise RuntimeError('completed two-build runtime ABBA required')
    if sha(folder/'benchmark_sd_runtime.py') != bench['benchmark_sha256']:
        raise RuntimeError('frozen benchmark source differs')
    for name, build in bench['builds'].items():
        path, manifest, digest = verify_snapshot(ROOT/build['snapshot'], name == 'candidate')
        if digest != build['manifest_sha256'] or manifest['sha256']['build/sd-baseline-ve/bin/sd'] != build['binary_sha256']:
            raise RuntimeError('build identity differs')
    proof = ROOT/'docs/results/20261009T093029Z-sd-turbo-nlc-unified.json'
    prior = json.loads(proof.read_text())
    if prior['binary_sha256'] != bench['builds']['candidate']['binary_sha256'] or prior['cpu_stage_checks'] != 62:
        raise RuntimeError('same-candidate six-prompt/four-step evidence required')
    prior_checks = 0
    for run in prior['runs']:
        native = (ROOT/run['native_folder']).resolve()
        native.relative_to(ROOT/'build')
        if sha(native/'summary.json') != run['summary_sha256'] or sha(native/'native.log') != run['native_log_sha256']:
            raise RuntimeError('earlier raw correctness artifacts changed')
        summary = json.loads((native/'summary.json').read_text())
        if not summary['completed'] or summary['binary_sha256'] != prior['binary_sha256']:
            raise RuntimeError('earlier candidate evidence incomplete')
        for request in summary['requests']:
            if not all(c['passed'] for c in request['checks']):
                raise RuntimeError('earlier CPU check failed')
            prior_checks += len(request['checks'])
            output = native/('request'+str(request['request']))
            for name, digest in request['trace_sha256'].items():
                if sha(output/name) != digest:
                    raise RuntimeError('earlier native trace changed')
            if sha(output/'image.png') != request['png_sha256']:
                raise RuntimeError('earlier PNG changed')
    if prior_checks != 62:
        raise RuntimeError('earlier check count differs')
    artifacts = {}
    checks = 0
    for index, run in enumerate(bench['runs']):
        native = (ROOT/run['artifacts']).resolve()
        native.relative_to(ROOT/'build')
        sampled = (ROOT/run['memory_artifacts']).resolve()
        sampled.relative_to(ROOT/'build')
        summary = json.loads((native/'summary.json').read_text())
        memory = json.loads((sampled/'summary.json').read_text())
        if not summary['completed'] or not memory['completed'] or summary['returncode'] or memory['final_used_kib'] != 131072:
            raise RuntimeError('completion and recovery evidence required')
        if summary['binary_sha256'] != bench['builds'][run['arm']]['binary_sha256'] or summary['checker_sha256'] != bench['checker_sha256'] or sha(native/'check_sd_resident.py') != bench['checker_sha256']:
            raise RuntimeError('executed binary/checker evidence differs')
        if summary['requests'] != run['requests'] or summary['process_seconds'] != run['process_seconds'] or memory != run['memory']:
            raise RuntimeError('aggregate differs from native evidence')
        log = (native/'native.log').read_text()
        times = [float(x) for x in re.findall(r'SD_REQUEST_END index=\d+ seconds=([0-9.]+)', log)]
        if len(times) != 2 or times != [r['request_seconds'] for r in run['requests']]:
            raise RuntimeError('native timer markers differ')
        for request in run['requests']:
            if len(request['checks']) != 5 or not all(c['passed'] for c in request['checks']):
                raise RuntimeError('all independent CPU stage checks required')
            checks += len(request['checks'])
            output = native/('request'+str(request['request']))
            for name, digest in request['trace_sha256'].items():
                if sha(output/name) != digest:
                    raise RuntimeError('native trace changed')
            if sha(output/'image.png') != request['png_sha256']:
                raise RuntimeError('PNG changed')
        samples = list(csv.DictReader((sampled/'memory.csv').open()))
        peak = max(int(row['used_kib']) for row in samples)
        if peak != memory['sampled_highest_used_kib'] or int(samples[-1]['used_kib']) != memory['final_used_kib']:
            raise RuntimeError('memory aggregate differs')
        artifacts[str(index)] = {str(p.relative_to(ROOT)): sha(p) for p in
                                 (folder/('run%d.log' % index), native/'summary.json', native/'native.log', sampled/'summary.json', sampled/'memory.csv')}
    recomputed = {}
    for label, index in (('process', None), ('cold', 0), ('hot', 1)):
        means = {name: sum(r['process_seconds'] if index is None else r['requests'][index]['request_seconds']
                          for r in bench['runs'] if r['arm'] == name)/2 for name in ('baseline', 'candidate')}
        recomputed[label] = {**means, 'latency_reduction_percent': (1-means['candidate']/means['baseline'])*100}
    if recomputed != bench['metrics']:
        raise RuntimeError('published metrics differ from recomputed means')
    equality = all(r['requests'][i]['trace_sha256'] == bench['runs'][0]['requests'][i]['trace_sha256'] and
                   r['requests'][i]['png_sha256'] == bench['runs'][0]['requests'][i]['png_sha256']
                   for r in bench['runs'] for i in (0, 1))
    if equality != bench['all_trace_and_png_bytes_identical']:
        raise RuntimeError('byte equality aggregate differs')
    stage_metrics = {}
    for stage in ('clip', 'unet', 'vae'):
        stage_metrics[stage] = {
            name: {part: sum(sum(p['seconds'] for p in r['requests'][1]['profile']
                                 if p['stage'] == stage and p['part'] == part)
                             for r in bench['runs'] if r['arm'] == name)/2
                   for part in ('backend_CPU', 'backend_BLAS', 'graph_compute')}
            for name in ('baseline', 'candidate')}
    thermal_artifacts = {}
    for name, log_path in (('abba', args.guard_log), ('build', args.build_log)):
        text = log_path.read_text()
        temperature_path = ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)', text)[-1]
        fan_path = ROOT/re.findall(r'Fan observation:.*log=([^\s]+)', text)[-1]
        rows = list(csv.DictReader(temperature_path.open()))
        if not rows or any(float(row['temperature_c']) >= float(row['stop_c']) for row in rows):
            raise RuntimeError('thermal stop threshold reached')
        thermal_artifacts[name] = {'sha256': {str(p.resolve().relative_to(ROOT)): sha(p) for p in (log_path, temperature_path, fan_path)},
                                   'warning_events': text.count('Temperature warning:'),
                                   'warning_samples': sum(float(r['temperature_c']) >= float(r['warn_c']) for r in rows)}
    report = {'status': 'faster_for_validated_workload' if recomputed['hot']['latency_reduction_percent'] > 0 else 'not_faster',
              'default_enabled': False, 'model': 'stabilityai/sd-turbo', 'runtime_slot': 1, 'precision': 'FP32', 'size': 512,
              'same_binary': False, 'metrics': recomputed, 'abba_cpu_stage_checks': checks,
              'hot_stage_profile_mean_seconds': stage_metrics,
              'earlier_same_candidate_cpu_stage_checks': 62, 'total_cpu_stage_checks': checks+62,
              'earlier_correctness_evidence': {'path': str(proof.relative_to(ROOT)), 'sha256': sha(proof)},
              'builds': bench['builds'], 'checker_sha256': bench['checker_sha256'], 'benchmark_sha256': bench['benchmark_sha256'],
              'publisher_sha256': sha(Path(__file__)), 'library_sha256': bench['library_sha256'],
              'all_trace_and_png_bytes_identical': equality,
              'runs': [{'arm': r['arm'], 'process_seconds': r['process_seconds'], 'request_seconds': [q['request_seconds'] for q in r['requests']],
                        'sampled_node_peak_gib': r['memory']['sampled_highest_used_kib']/1048576,
                        'final_node_used_mib': r['memory']['final_used_kib']/1024} for r in bench['runs']],
              'abba_temperature': thermal(args.guard_log), 'build_temperature': thermal(args.build_log),
              'abba_fan_detail': fan_detail(args.guard_log), 'artifact_sha256': artifacts,
              'thermal_artifact_evidence': thermal_artifacts,
              'scope': 'one fixed prompt, one Euler step, two runs per arm, two requests per process; candidate separately checked on six prompts and four steps',
              'limitations': ['generic runtime and NLC jointly changed; not a same-binary or single-factor comparison',
                             'includes profiling and thread diagnostics; not a production service benchmark',
                             'two runs per arm are insufficient to characterize tails or long-run variance',
                             'whole-node memory is discrete, not instantaneous or model-exclusive',
                             'unchanged readable fan states do not prove physical fan strategy unchanged']}
    output = args.output.resolve()
    output.relative_to(ROOT/'docs/results')
    safe(report)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Audited runtime result:', output.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
