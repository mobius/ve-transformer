"""Audit current selective-thread operator timings; no speedup inference."""
import argparse
import csv
import math
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
from record_sd_softmax_model import softmax_dispatch

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
    baseline=ROOT/'build/accepted/softmax-scale-abba-20261009T210240Z/build/sd-baseline-ve/manifest.json'
    previous=json.loads(baseline.read_text())
    if accepted.get('status')!='formal_softmax_scale_abba_verified' or not accepted['all_trace_and_png_bytes_identical'] or sha(baseline)!=accepted['model_manifest_sha256'] or accepted['checker_sha256']!=value['checker_sha256']:raise RuntimeError('accepted current formal model required')
    added=set(manifest['sha256'])-set(previous['sha256']);removed=set(previous['sha256'])-set(manifest['sha256'])
    changed=[n for n,h in manifest['sha256'].items() if n in previous['sha256'] and previous['sha256'][n]!=h]
    if removed or added!={'scripts/prepare_sd_group_norm_profile.py'} or set(changed)!={'scripts/prepare_sd_baseline_overlay.py','scripts/record_sd_baseline_build.py','build/sd-baseline-overlay/sd-ggml-cpu.c','build/sd-baseline-ve/bin/sd'}:raise RuntimeError('only diagnostic inputs may change')
    from prepare_sd_group_norm_profile import instrument
    if (ROOT/'build/sd-baseline-overlay/sd-ggml-cpu.c').read_text()!=instrument((baseline.parent.parent/'sd-baseline-overlay/sd-ggml-cpu.c').read_text()):raise RuntimeError('CPU source differs beyond diagnostic instrumentation')

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
    pattern=r'SD_GROUP_NORM_SHAPE stage=(clip|unet|vae) threads=(\d+) src_type=(\d+) dst_type=(\d+) ne=([0-9,]+) src_nb=([0-9,]+) dst_nb=([0-9,]+) groups=(\d+) eps=([^\s]+)'
    for request,body in zip(value['requests'],segments):
        found=re.findall(pattern,body);shapes=[]
        if len(found)!=body.count('SD_GROUP_NORM_SHAPE '):raise RuntimeError('complete GroupNorm metadata required')
        for occurrence,(stage,nth,st,dt,ne,snb,dnb,groups,eps) in enumerate(found):
            ne=list(map(int,ne.split(',')));snb=list(map(int,snb.split(',')));dnb=list(map(int,dnb.split(',')));groups=int(groups);epsilon=float(eps)
            if len(ne)!=4 or len(snb)!=4 or len(dnb)!=4 or any(n<=0 for n in ne) or st!='0' or dt!='0' or nth!='8' or not 0<groups<=ne[2] or not math.isfinite(epsilon) or epsilon<=0 or snb[0]!=4 or dnb!=[4,4*ne[0],4*ne[0]*ne[1],4*ne[0]*ne[1]*ne[2]]:raise RuntimeError('valid actual F32 GroupNorm geometry required')
            shapes.append(dict(stage=stage,occurrence=occurrence,threads=int(nth),ne=ne,src_nb=snb,dst_nb=dnb,groups=groups,epsilon=epsilon))
        for stage in ('clip','unet','vae'):
            total=sum(o['nodes'] for o in request['operator_profile'] if o['stage']==stage and o['op']=='GROUP_NORM')
            if len([r for r in shapes if r['stage']==stage])!=total:raise RuntimeError('GroupNorm metadata/operator counts differ')
        if not shapes:raise RuntimeError('actual GroupNorm shapes absent')
        shape_requests.append(dict(shapes=shapes))
    if len(shape_requests)!=2 or shape_requests[0]!=shape_requests[1]:raise RuntimeError('cold and hot GroupNorm geometry differs')
    scale_dispatch=softmax_dispatch(native,value)
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
    report = {'cpu_recomputed_checks':cpu_checks,'cpu_recomputed_png_checks':png_count,'cpu_reference_sha256':reference_sha,'cont_dispatch':dispatch,'accepted_proof_sha256':sha(proof),'shape_requests':shape_requests,'softmax_scale_dispatch':scale_dispatch,'changed_build_inputs':changed,'added_build_inputs':sorted(added),'status':'group_norm_shape_profile_audited','binary_sha256':value['binary_sha256'],
              'checker_sha256':value['checker_sha256'],'publisher_sha256':sha(Path(__file__)),
              'audit_helper_sha256':{name:sha(ROOT/name) for name in ('scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_pixels_pack_model.py','scripts/record_sd_cont_extended_model.py','scripts/record_sd_softmax_model.py','scripts/prepare_sd_group_norm_profile.py')},
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
    print('GroupNorm shape profile audited:',output.relative_to(ROOT))


if __name__ == '__main__':
    main()
