"""Audit the GELU ABBA against native traces and earlier checks."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail

ROOT = Path(__file__).resolve().parents[1]


def check_requests(folder, value, binary, checker=None):
    if not value['completed'] or value['binary_sha256'] != binary:
        raise RuntimeError('completed same-candidate request evidence required')
    if sha(folder/'check_sd_resident.py') != value['checker_sha256']:
        raise RuntimeError('frozen checker changed')
    if checker is not None and value['checker_sha256'] != checker:
        raise RuntimeError('ABBA checker differs')
    text = (folder/'native.log').read_text()
    times = [float(t) for t in re.findall(r'SD_REQUEST_END index=\d+ seconds=([0-9.]+)', text)]
    if times != [r['request_seconds'] for r in value['requests']]:
        raise RuntimeError('native request timers differ')
    count = 0
    segments = re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+', text, re.S)
    if len(segments) != len(value['requests']):
        raise RuntimeError('ordered native request segments required')
    for request, body in zip(value['requests'], segments):
        dispatch = [{'stage': a, 'op': b, 'optimized': int(c), 'fallback': int(d), 'enabled': bool(int(e))}
                    for a,b,c,d,e in re.findall(r'SD_BINARY_DISPATCH stage=(clip|unet|vae) op=(ADD|MUL) optimized=(\d+) fallback=(\d+) enabled=(0|1)', body)]
        if dispatch != request['binary_dispatch']:
            raise RuntimeError('dispatch evidence differs from native log')
        switches = [{'stage': a, 'before': int(b), 'configured': int(c), 'after': int(d), 'seconds': float(e)}
                    for a,b,c,d,e in re.findall(r'SD_STAGE_THREADS stage=(clip|unet|vae) before=(\d+) configured=(\d+) after=(\d+) seconds=([0-9.]+)', body)]
        if switches != request['stage_thread_switches']:
            raise RuntimeError('stage switch evidence differs from native log')
        profiles = [{'stage': a, 'part': b, 'seconds': float(c), 'calls': int(d)}
                    for a,b,c,d in re.findall(r'SD_PROFILE stage=(clip|unet|vae) part=([a-zA-Z_]+) seconds=([0-9.]+) calls=(\d+)', body)]
        if profiles != request['profile']:
            raise RuntimeError('component timing differs from native log')
        gelu = [{'stage': a, 'optimized': int(b), 'fallback': int(c), 'enabled': bool(int(d))}
                for a,b,c,d in re.findall(r'SD_GELU_DISPATCH stage=(clip|unet|vae) optimized=(\d+) fallback=(\d+) enabled=(0|1)', body)]
        if gelu != request['gelu_dispatch']:
            raise RuntimeError('GELU dispatch differs from native log')
        for stage, nodes in (('clip',23),('unet',16),('vae',0)):
            entries = [r for r in gelu if r['stage']==stage]
            wanted = (nodes,0,True) if value['gelu_mode']=='ve' else (0,nodes,False)
            if len(entries)!=(value['steps'] if stage=='unet' else 1) or any((r['optimized'],r['fallback'],r['enabled'])!=wanted for r in entries):
                raise RuntimeError('actual GELU dispatch count differs')

        generic = re.findall(r'SD_GGML_THREADS stage=(clip|unet|vae) actual=(\d+) maximum=(\d+)', body)
        nlc = re.findall(r'SD_NLC_THREADS stage=(clip|unet|vae) configured=(\d+) actual=(\d+) idle_after=(\d+)', body)
        for stage in ('clip','unet','vae'):
            cpu_threads = value['vae_threads'] if stage=='vae' else 8
            blas_threads = (value['vae_blas_threads'] or value['vae_threads']) if stage=='vae' else 8
            idle = 8 if value['vae_blas_threads']==4 else cpu_threads
            cpu_rows = [r for r in generic if r[0]==stage]
            blas_rows = [r for r in nlc if r[0]==stage]
            if (len(cpu_rows)!=sum(p['calls'] for p in profiles if p['stage']==stage and p['part']=='backend_CPU') or
                    any(tuple(map(int,r[1:]))!=(cpu_threads,cpu_threads) for r in cpu_rows) or
                    len(blas_rows)!=sum(p['calls'] for p in profiles if p['stage']==stage and p['part']=='backend_BLAS') or
                    any(tuple(map(int,r[1:]))!=(blas_threads,blas_threads,idle) for r in blas_rows)):
                raise RuntimeError('native actual team or restore differs')

        if len(request['checks']) != 3+2*value['steps'] or not all(c['passed'] for c in request['checks']):
            raise RuntimeError('independent CPU stage checks required')
        count += len(request['checks'])
        output = folder/('request'+str(request['request']))
        for name, digest in request['trace_sha256'].items():
            if sha(output/name) != digest:
                raise RuntimeError('native trace changed')
        if sha(output/'image.png') != request['png_sha256']:
            raise RuntimeError('native PNG changed')
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--guard-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.benchmark.resolve()
    folder.relative_to(ROOT/'build')
    bench = json.loads((folder/'summary.json').read_text())
    if bench['kind'] != 'sd_turbo_gelu_abba' or not bench['completed'] or [r['mode'] for r in bench['runs']] != ['generic','ve','ve','generic']:
        raise RuntimeError('complete GELU ABBA required')
    if sha(folder/'benchmark_sd_resident.py') != bench['benchmark_sha256']:
        raise RuntimeError('frozen benchmark changed')
    manifest_path = ROOT/'build/sd-baseline-ve/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('build source or executable changed')
    if manifest['generic_thread_runtime'] != 'openmp' or manifest['nlc_mode'] != 'openmp' or manifest['sha256']['build/sd-baseline-ve/bin/sd'] != bench['binary_sha256']:
        raise RuntimeError('same unified OpenMP build required')
    prior_path = ROOT/'docs/results/20261009T115851Z-sd-turbo-gelu-model.json'
    prior = json.loads(prior_path.read_text())
    earlier_checks = 0
    for run in prior['tests']:
        native_summary = ROOT/run['summary']
        if sha(native_summary) != run['summary_sha256'] or sha(ROOT/run['log']) != run['log_sha256']:
            raise RuntimeError('earlier raw evidence changed')
        memory_path = ROOT/run['memory_summary']
        if sha(memory_path) != run['memory_summary_sha256']:
            raise RuntimeError('earlier memory evidence changed')
        value = json.loads(native_summary.read_text())
        if value['gelu_mode'] != 've' or value['vae_threads'] != 8 or value['vae_blas_threads'] != 4 or value['checker_sha256'] != bench['checker_sha256']:
            raise RuntimeError('matching earlier selective VAE checks required')
        earlier_checks += check_requests(native_summary.parent, value, bench['binary_sha256'], bench['checker_sha256'])
    if earlier_checks != 62 or prior['total_independent_cpu_checks'] != earlier_checks:
        raise RuntimeError('earlier CPU stage count differs')
    if prior['actual_model_object_kernel_fixtures'] != 102:
        raise RuntimeError('actual model object boundary proof required')
    objects = list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('ve_sd_turbo_gelu.c.o'))
    if len(objects)!=1 or sha(objects[0])!=prior['actual_model_object_sha256']:
        raise RuntimeError('model kernel object differs from tested object')
    for name, digest in prior['artifact_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError('kernel/model validation evidence changed')
    checks = 0
    evidence = {}
    for index, run in enumerate(bench['runs']):
        native = (ROOT/run['artifacts']).resolve()
        native.relative_to(ROOT/'build')
        sampled = (ROOT/run['memory_artifacts']).resolve()
        sampled.relative_to(ROOT/'build')
        value = json.loads((native/'summary.json').read_text())
        if value['steps'] != 1 or len(value['requests']) != 2 or value['reference_artifacts'] != bench['reference_artifacts']:
            raise RuntimeError('same one-step frozen reference and two requests required')
        if value['binary_scalar_mode'] != 've' or value['vae_threads'] != 8 or value['vae_blas_threads'] != 4 or value['gelu_mode'] != run['mode'] or value['operator_profile_enabled'] or value['binary_shape_profile_enabled'] or value['nlc_threads'] != 'unified' or value['mode'] != 'resident' or value['tokenizer_mode'] != 'resident':
            raise RuntimeError('fixed comparison configuration differs')
        checks += check_requests(native, value, bench['binary_sha256'], bench['checker_sha256'])
        if value['requests'] != run['requests'] or value['process_seconds'] != run['process_seconds']:
            raise RuntimeError('aggregate differs from original requests')
        memory = json.loads((sampled/'summary.json').read_text())
        if not memory['completed'] or memory != run['memory'] or memory['final_used_kib'] != 131072:
            raise RuntimeError('memory completion/recovery required')
        rows = list(csv.DictReader((sampled/'memory.csv').open()))
        if max(int(r['used_kib']) for r in rows) != memory['sampled_highest_used_kib'] or int(rows[-1]['used_kib']) != 131072:
            raise RuntimeError('memory CSV differs')
        files = [folder/('run%d.log' % index), native/'summary.json', native/'native.log', native/'check_sd_resident.py', sampled/'summary.json', sampled/'memory.csv']
        evidence[str(index)] = {str(p.relative_to(ROOT)): sha(p) for p in files}
    if checks != 40:
        raise RuntimeError('forty ABBA CPU checks required')
    metrics = {}
    for name, index in (('process', None), ('request0', 0), ('request1', 1)):
        means = {mode: sum(r['process_seconds'] if index is None else r['requests'][index]['request_seconds']
                           for r in bench['runs'] if r['mode'] == mode)/2 for mode in ('generic','ve')}
        metrics[name] = {'generic_seconds': means['generic'], 've_seconds': means['ve'],
                         'latency_reduction_percent': (1-means['ve']/means['generic'])*100}
    if metrics != bench['metrics']:
        raise RuntimeError('metrics differ from recomputed means')
    equality = all(r['requests'][i]['trace_sha256'] == bench['runs'][0]['requests'][i]['trace_sha256'] and
                   r['requests'][i]['png_sha256'] == bench['runs'][0]['requests'][i]['png_sha256']
                   for r in bench['runs'] for i in (0,1))
    if equality != bench['all_trace_and_png_bytes_identical']:
        raise RuntimeError('trace/PNG equality record differs')
    trace_differences = []
    image_differences = []
    for run in bench['runs']:
        per_request = []
        per_image = []
        for index in (0,1):
            reference = ROOT/bench['runs'][0]['artifacts']/('request'+str(index))
            candidate = ROOT/run['artifacts']/('request'+str(index))
            differences = {}
            for name in bench['runs'][0]['requests'][index]['trace_sha256']:
                if name.endswith('.f32'):
                    a=np.fromfile(reference/name,dtype='<f4').astype(np.float64)
                    b=np.fromfile(candidate/name,dtype='<f4').astype(np.float64)
                    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise RuntimeError('matching finite F32 output traces required')
                    delta=b-a
                    differences[name]={'max_abs_error':float(np.max(np.abs(delta))),
                                       'relative_l2':float(np.sqrt(np.sum(delta*delta)/max(np.sum(a*a),1e-30)))}
                elif sha(reference/name)!=sha(candidate/name):
                    raise RuntimeError('non-F32 input/metadata traces must match')
            with Image.open(reference/'image.png') as image:
                if image.mode!='RGB' or image.size!=(512,512):raise RuntimeError('fixed RGB image required')
                a=np.array(image,dtype=np.int16)
            with Image.open(candidate/'image.png') as image:
                if image.mode!='RGB' or image.size!=(512,512):raise RuntimeError('fixed RGB image required')
                b=np.array(image,dtype=np.int16)
            per_image.append({'changed_channels':int(np.count_nonzero(a!=b)),
                              'changed_pixels':int(np.count_nonzero(np.any(a!=b,axis=2))),
                              'max_abs_byte_delta':int(np.max(np.abs(a-b)))})
            per_request.append(differences)
        trace_differences.append({'mode':run['mode'],'requests':per_request})
        image_differences.append({'mode':run['mode'],'requests':per_image})
    guard_text = args.guard_log.read_text()
    temperature = ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)', guard_text)[-1]
    fans = ROOT/re.findall(r'Fan observation:.*log=([^\s]+)', guard_text)[-1]
    thermal_rows = list(csv.DictReader(temperature.open()))
    if not thermal_rows or any(float(r['temperature_c']) >= float(r['stop_c']) for r in thermal_rows):
        raise RuntimeError('thermal stop threshold reached')
    stage_means = {stage: {mode: {part: sum(sum(p['seconds'] for p in r['requests'][1]['profile'] if p['stage']==stage and p['part']==part)
                                           for r in bench['runs'] if r['mode']==mode)/2
                                 for part in ('backend_CPU','backend_BLAS','graph_compute','nlc_thread_prepare','nlc_thread_release')}
                          for mode in ('generic','ve')} for stage in ('clip','unet','vae')}
    report = {'status': 'faster_for_validated_workload' if metrics['request1']['latency_reduction_percent'] > 0 else 'not_faster',
              'default_enabled': False, 'same_binary': True, 'model': 'stabilityai/sd-turbo', 'precision': 'FP32', 'size': 512, 'runtime_slot': 1,
              'binary_sha256': bench['binary_sha256'], 'checker_sha256': bench['checker_sha256'], 'benchmark_sha256': bench['benchmark_sha256'],
              'publisher_sha256': sha(Path(__file__)), 'build_manifest_sha256': sha(manifest_path),
              'png_pixel_differences_vs_first_baseline':image_differences, 'trace_differences_vs_first_baseline':trace_differences, 'metrics': metrics, 'hot_stage_profile_mean_seconds': stage_means, 'all_trace_and_png_bytes_identical': equality, 'all_png_bytes_identical': all(r['requests'][i]['png_sha256']==bench['runs'][0]['requests'][i]['png_sha256'] for r in bench['runs'] for i in (0,1)),
              'actual_model_object_kernel_fixtures':102, 'abba_cpu_stage_checks': checks, 'earlier_cpu_stage_checks': earlier_checks, 'total_cpu_stage_checks': checks+earlier_checks,
              'gelu_abba': ['generic','ve','ve','generic'], 'earlier_proof': {'path': str(prior_path.relative_to(ROOT)), 'sha256': sha(prior_path)},
              'runs': [{'mode': r['mode'], 'process_seconds': r['process_seconds'], 'request_seconds': [q['request_seconds'] for q in r['requests']],
                        'cold_weights_load_seconds': sum(p['seconds'] for p in r['requests'][0]['profile'] if p['part']=='weights_load'),
                        'sampled_node_peak_gib': r['memory']['sampled_highest_used_kib']/1048576, 'final_node_used_mib':128} for r in bench['runs']],
              'temperature': thermal(args.guard_log), 'fan_detail': fan_detail(args.guard_log), 'warning_events':guard_text.count('Temperature warning:'),
              'artifact_sha256': evidence, 'thermal_artifact_sha256': {str(p.resolve().relative_to(ROOT)):sha(p) for p in (args.guard_log,temperature,fans)},
              'scope': 'two runs per mode, two same-process requests, one fixed one-step prompt; six-prompt/four-step candidate correctness separately verified',
              'limitations': ['includes stage/thread/dispatch diagnostics and native output IO; not a production service benchmark',
                             'checked vector erf may change rounding; independent CPU reference tolerances remain required',
                             'cold loading caches not controlled; two repeats do not characterize tails or long-run variance',
                             'other sizes, models, precisions and concurrent requests not covered',
                             'memory is discrete whole-node usage; readable fan states do not establish actual physical fan policy']}
    safe(report)
    output = args.output.resolve()
    output.relative_to(ROOT/'docs/results')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Audited GELU result:', output.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
