"""Audit current selective-thread operator timings; no speedup inference."""
import argparse
import csv
import math
from collections import defaultdict
import json
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch
from record_sd_pixels_pack_model import recompute_cpu_reference
from record_sd_cont_extended_model import cont_dispatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof',type=Path,required=True)
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
    proof=args.proof.resolve();proof.relative_to(ROOT/'docs/results');accepted=json.loads(proof.read_text())
    baseline=ROOT/'build/accepted/cont-extended-abba-20261009T195919Z/build/sd-baseline-ve/manifest.json'
    prior_manifest=json.loads(baseline.read_text())
    if accepted.get('status')!='formal_cont_extended_abba_verified' or not accepted['all_trace_and_png_bytes_identical'] or sha(baseline)!=accepted['model_manifest_sha256'] or accepted['checker_sha256']!=value['checker_sha256']:raise RuntimeError('accepted baseline required')
    allowed={'scripts/prepare_sd_baseline_overlay.py','build/sd-baseline-overlay/sd-ggml-cpu.c','build/sd-baseline-ve/bin/sd'}
    if set(manifest['sha256'])!=set(prior_manifest['sha256']):raise RuntimeError('manifest inputs differ')
    changed=[n for n,h in manifest['sha256'].items() if h!=prior_manifest['sha256'][n]]
    if set(changed)!=allowed:raise RuntimeError('only diagnostic generator/source/binary may change')
    if value.get('vae_spatial_tile')!='8192' or value.get('png_encoder')!='ve' or value.get('rgb_buffer')!='resident' or value.get('pixel_kernel')!='ve' or value.get('image_output_profile_enabled'):raise RuntimeError('matching current complete configuration required')
    dispatch=cont_dispatch(native,value,'extended')
    cpu_checks,png_count,reference_sha=recompute_cpu_reference(native,value)
    if len(cpu_checks)!=10 or png_count!=2:raise RuntimeError('full diagnostic CPU/PNG audit required')
    checks = check_requests(native, value, value['binary_sha256'])
    rows_dispatch(native, value, 'rows_256')
    raw = (native/'native.log').read_text()
    segments = re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+', raw, re.S)
    for request, body in zip(value['requests'], segments):
        operators = [{'stage': a, 'op': b, 'seconds': float(c), 'nodes': int(d)}
                     for a,b,c,d in re.findall(r'SD_OP_PROFILE stage=(clip|unet|vae) op=([A-Z_0-9]+) seconds=([0-9.]+) nodes=(\d+)', body)]
        if operators != request['operator_profile']:
            raise RuntimeError('operator timings differ from raw log')
    shape_requests=[]
    for request,body in zip(value['requests'],segments):
        shapes=[];phases=[]
        for line in body.splitlines():
            if line.startswith('SD_SOFTMAX_SHAPE ') or line.startswith('SD_SOFTMAX_PHASE '):
                pairs=[x.split('=',1) for x in line.split()[1:]];row=dict(pairs)
                if len(row)!=len(pairs):raise RuntimeError('duplicate diagnostic fields')
                if line.startswith('SD_SOFTMAX_SHAPE '):
                    if set(row)!={'stage','node','threads','type','ne','nb','rows','columns','mask_type','scale','bias','seconds'}:raise RuntimeError('complete shape metadata required')
                    for k in ('node','threads','type','rows','columns','mask_type'):row[k]=int(row[k])
                    for k in ('ne','nb'):row[k]=[int(x) for x in row[k].split(',')]
                    for k in ('scale','bias','seconds'):row[k]=float(row[k])
                    if row['stage'] not in ('clip','unet','vae') or row['threads']!=8 or row['type']!=0 or row['mask_type'] not in (-1,0,1) or len(row['ne'])!=4 or len(row['nb'])!=4 or any(x<1 for x in row['ne']) or row['nb'][0]!=4 or row['columns']!=row['ne'][0] or row['rows']!=math.prod(row['ne'][1:]) or any(not math.isfinite(row[k]) for k in ('scale','bias','seconds')) or row['seconds']<0:raise RuntimeError('valid actual shape required')
                    row['occurrence']=len(shapes)
                    shapes.append(row)
                else:
                    if set(row)!={'stage','node','thread','rows','prepare_us','max_us','exp_us','sum_us','normalize_us'}:raise RuntimeError('complete phase metadata required')
                    for k in set(row)-{'stage'}:row[k]=int(row[k])
                    if any(row[k]<0 for k in set(row)-{'stage'}):raise RuntimeError('nonnegative phase counters required')
                    if not shapes or (row['stage'],row['node'])!=(shapes[-1]['stage'],shapes[-1]['node']):raise RuntimeError('phase must follow its node occurrence')
                    row['occurrence']=len(shapes)-1
                    phases.append(row)
        if not shapes or len(phases)!=8*len(shapes):raise RuntimeError('all unique nodes and workers required')
        for r in shapes:
            rows=[x for x in phases if x['occurrence']==r['occurrence']];dr=(r['rows']+7)//8
            if [x['thread'] for x in rows]!=list(range(8)) or [x['rows'] for x in rows]!=[max(0,min(dr*(i+1),r['rows'])-dr*i) for i in range(8)]:raise RuntimeError('actual worker row partition differs')
        for stage in ('clip','unet','vae'):
            ops=[x for x in request['operator_profile'] if x['stage']==stage and x['op']=='SOFT_MAX'];nodes=[r for r in shapes if r['stage']==stage]
            if len(ops)!=1 or len(nodes)!=ops[0]['nodes'] or abs(sum(r['seconds'] for r in nodes)-ops[0]['seconds'])>1e-9:raise RuntimeError('node timing sum differs from operator aggregate')
        shape_requests.append(dict(shapes=shapes,phases=phases))
    if [[{k:v for k,v in r.items() if k!='seconds'} for r in x['shapes']] for x in shape_requests][0]!=[{k:v for k,v in r.items() if k!='seconds'} for r in shape_requests[1]['shapes']]:raise RuntimeError('cold/hot geometry differs')
    phase_work={stage:{k:sum(r[k] for r in shape_requests[1]['phases'] if r['stage']==stage) for k in ('prepare_us','max_us','exp_us','sum_us','normalize_us')} for stage in ('clip','unet','vae')}
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
    files = [proof,baseline,manifest_path]+list(native.rglob('*.f32'))+list(native.rglob('*.png'))+[native/'summary.json',native/'native.log',native/'check_sd_resident.py',
             memory/'summary.json',memory/'memory.csv',args.guard_log,temperature,fans]
    report = {'cpu_recomputed_checks':cpu_checks,'cpu_recomputed_png_checks':png_count,'cpu_reference_sha256':reference_sha,'cont_dispatch':dispatch,'accepted_proof_sha256':sha(proof),'shape_requests':shape_requests,'hot_phase_worker_us':phase_work,'changed_build_inputs':changed,'status':'softmax_shape_profile_audited','binary_sha256':value['binary_sha256'],
              'checker_sha256':value['checker_sha256'],'publisher_sha256':sha(Path(__file__)),
              'audit_helper_sha256':{name:sha(ROOT/name) for name in ('scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_pixels_pack_model.py','scripts/record_sd_cont_extended_model.py')},
              'im2col_mode':'rows_256','gelu_mode':'ve',
              'manifest_sha256':sha(manifest_path),'cpu_stage_checks':checks,
              'request_seconds':[r['request_seconds'] for r in value['requests']],
              'hot_stages':stages,'hot_component_profile':hot['profile'],
              'temperature':thermal(args.guard_log),'fan_detail':fan_detail(args.guard_log),
              'sampled_node_peak_gib':sampled['sampled_highest_used_kib']/1048576,
              'final_node_used_mib':128,
              'artifact_sha256':{str(p.resolve().relative_to(ROOT)):sha(p) for p in files},
              'scope':'one fixed one-step prompt, cold and hot request; includes operator timing and barriers; diagnostic only; per-row microsecond timers add overhead, summed worker phase time is not node latency; no speedup claim'}
    safe(report)
    output = args.output.resolve()
    output.relative_to(ROOT/'docs/results')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Softmax shape/phase profile audited:',output.relative_to(ROOT))


if __name__ == '__main__':
    main()
