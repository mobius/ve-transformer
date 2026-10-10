"""Audit actual mixed spatial blocks, retained objects and full CPU/PNG outputs."""
import argparse,csv,json,os,re,subprocess
from collections import Counter
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_model import recompute_cpu_reference
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch
from record_sd_png_model import png_dispatch,png_compiler_evidence
from record_sd_pixels_pack_abba import guarded_csv
from record_sd_group_norm_model import cont_dispatch,compiler_evidence,softmax_dispatch,softmax_compiler_evidence,group_norm_dispatch,group_norm_compiler_evidence
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
from record_sd_gemm_untiled_model import tile_dispatch
from record_sd_current_vae_build import evidence as current_build_evidence,retained_compilers
def audited_run(run,memory,guard,binary,checker,steps,cases,mode='mixed2048',strategy='all4'):
 if strategy not in ('all4','selective'):raise RuntimeError('explicit thread strategy required')
 value=json.loads((run/'summary.json').read_text())
 fixed={'mode':'resident','tokenizer_mode':'resident','nlc_threads':'unified','nlc_mode':'openmp','vae_threads':4 if strategy=='all4' else 8,'vae_blas_threads':None if strategy=='all4' else 4,'gelu_mode':'ve','binary_scalar_mode':'ve','im2col_mode':'rows_256','vae_spatial_tile':mode,'png_encoder':'ve','rgb_buffer':'resident','pixel_kernel':'ve'}
 if any(value.get(k)!=v for k,v in fixed.items()) or value['steps']!=steps or [r['reference_case'] for r in value['requests']]!=cases or any(value.get(k) for k in ('operator_profile_enabled','binary_shape_profile_enabled','image_output_profile_enabled')):raise RuntimeError('fixed optimized workload differs')
 checks=check_requests(run,value,binary,checker);rows_dispatch(run,value,'rows_256');tile_dispatch(run,value,mode);png_dispatch(run,value,'ve')
 raw=(run/'native.log').read_text()
 if any(marker in raw for marker in ('SD_NLC_GEMM ','SD_IM2COL_SHAPE ','SD_IMAGE_OUTPUT_PROFILE ')):raise RuntimeError('unexpected detailed timing probes')
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S);image_times=[]
 for index,(body,request) in enumerate(zip(bodies,value['requests'])):
  switches=re.findall(r'SD_STAGE_THREADS stage=(clip|unet|vae) before=(\d+) configured=(\d+) after=(\d+) seconds=([0-9.]+)',body)
  if strategy=='all4':
   expected=[('clip',4 if index else 8,8,8)]+[('unet',8,8,8)]*steps+[('vae',8,4,4)]
   if [(r[0],)+tuple(map(int,r[1:4])) for r in switches]!=expected or len(switches)!=body.count('SD_STAGE_THREADS '):raise RuntimeError('actual stage switch state differs')
  elif switches or body.count('SD_STAGE_THREADS '):raise RuntimeError('selective strategy must keep stage capacity eight')
  reused=int(index>0);rgb=re.findall(r'SD_RGB_BUFFER mode=(request|resident) allocated=(0|1) reused=(0|1) bytes=(\d+)',body)
  pixels=re.findall(r'SD_PIXEL_KERNEL mode=(generic|ve) clamp_calls=(\d+) pack_calls=(\d+) spatial=(\d+) bytes=(\d+)',body)
  if rgb!=[('resident',str(1-reused),str(reused),'786432')] or body.count('SD_RGB_BUFFER ')!=1 or request['rgb_buffer_dispatch']!=dict(mode='resident',allocated=1-reused,reused=reused,bytes=786432) or pixels!=[('ve','1','1','262144','786432')] or body.count('SD_PIXEL_KERNEL ')!=1 or request['pixel_kernel_dispatch']!=dict(mode='ve',clamp_calls=1,pack_calls=1,spatial=262144,bytes=786432):raise RuntimeError('actual RGB/pixel dispatch differs')
  images=re.findall(r'SD_PROFILE stage=pipeline part=image_output seconds=([0-9.]+) calls=(\d+)',body)
  if len(images)!=1 or images[0][1]!='1' or not 0<float(images[0][0])<request['request_seconds']:raise RuntimeError('bounded unique output timer required')
  image_times.append(float(images[0][0]))
 recomputed,png_count,reference_sha=recompute_cpu_reference(run,value)
 if len(recomputed)!=checks:raise RuntimeError('CPU audit count differs')
 sampled=json.loads((memory/'summary.json').read_text());rows=list(csv.DictReader((memory/'memory.csv').open()))
 if not sampled['completed'] or sampled['returncode'] or sampled['final_used_kib']!=131072 or not rows or int(rows[-1]['used_kib'])!=131072 or max(int(r['used_kib']) for r in rows)!=sampled['sampled_highest_used_kib']:raise RuntimeError('memory completion/recovery differs')
 files=[run/'summary.json',run/'native.log',run/'check_sd_resident.py',memory/'summary.json',memory/'memory.csv',guard]+guarded_csv(guard)
 for request in value['requests']:
  output=run/('request'+str(request['request']));files.extend(output/name for name in request['trace_sha256']);files.append(output/'image.png')
 entry=dict(packing_input_domain_verified=True,artifacts=str(run.relative_to(ROOT)),memory_artifacts=str(memory.relative_to(ROOT)),request_seconds=[r['request_seconds'] for r in value['requests']],process_seconds=value['process_seconds'],image_output_seconds=image_times,cpu_checks=checks,cpu_recomputed_checks=recomputed,cpu_recomputed_png_checks=png_count,temperature=thermal(guard),fan_detail=fan_detail(guard),sampled_node_peak_gib=sampled['sampled_highest_used_kib']/1048576,sha256={str(p.relative_to(ROOT)):sha(p) for p in files},cpu_reference_sha256=reference_sha)
 return value,entry

def main():
 if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':raise RuntimeError('temperature supervision required')
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');paths=[x.resolve() for x in (a.run,a.memory,a.guard_log)]
 for path in paths:path.relative_to(ROOT/'build')
 manifest_path=ROOT/'build/sd-baseline-ve/manifest.json';manifest=json.loads(manifest_path.read_text())
 for name,digest in manifest['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('current model input changed')
 binary=sha(ROOT/'build/sd-baseline-ve/bin/sd');checker=sha(ROOT/'tests/check_sd_resident.py');steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[a.name]
 accepted_path=ROOT/'docs/results/20261010T000834Z-sd-turbo-gemm-untiled-abba.json';accepted=json.loads(accepted_path.read_text())
 if accepted['status']!='formal_untiled_spatial_abba_verified' or not accepted.get('cpu_validation_temperature') or accepted['checker_sha256']!=checker:raise RuntimeError('accepted previous model/checker required')
 build_evidence=current_build_evidence()
 result,entry=audited_run(*paths,binary,checker,steps,cases,'mixed2048','all4')
 entry.update(name=a.name,cont_dispatch=cont_dispatch(paths[0],result,'extended'),softmax_scale_dispatch=softmax_dispatch(paths[0],result),group_norm_dispatch=group_norm_dispatch(paths[0],result))
 compilers=retained_compilers()
 value=json.loads(proof.read_text()) if proof.exists() else dict(status='model_validation_in_progress',tests=[],model_manifest_sha256=sha(manifest_path),binary_sha256=binary,checker_sha256=checker)
 if value['model_manifest_sha256']!=sha(manifest_path) or value['binary_sha256']!=binary or value['checker_sha256']!=checker or any(value.get(k,v)!=v for k,v in compilers.items()):raise RuntimeError('model/checker/actual objects changed')
 existing=[e for e in value['tests'] if e['name']!=a.name]
 for e in existing:
  for group in ('sha256','cpu_reference_sha256'):
   for name,digest in e[group].items():
    if sha(ROOT/name)!=digest:raise RuntimeError('earlier evidence changed')
 value.update(compilers);value['tests']=sorted(existing+[entry],key=lambda e:('double','six','four').index(e['name']));value['completed_tests']=[e['name'] for e in value['tests']];value['all_candidate_tests_completed']=value['completed_tests']==['double','six','four'];value['status']='model_validation_verified' if value['all_candidate_tests_completed'] else 'model_validation_in_progress';value['independent_cpu_checks']=sum(e['cpu_checks'] for e in value['tests']);value['cpu_recomputed_checks']=sum(len(e['cpu_recomputed_checks']) for e in value['tests']);value['cpu_recomputed_png_checks']=sum(e['cpu_recomputed_png_checks'] for e in value['tests']);value['full_graph_speedup_measured']=False
 value['publisher_sha256']=sha(Path(__file__));value['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_gemm_untiled_model.py','scripts/record_sd_group_norm_model.py','scripts/record_sd_pixels_pack_model.py','scripts/record_sd_pixels_pack_abba.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_png_model.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/prepare_sd_group_norm_model.py','scripts/prepare_sd_gemm_spatial_small_model.py','scripts/prepare_sd_gemm_untiled_model.py','scripts/benchmark_sd_runtime.py','scripts/record_sd_current_vae_build.py','scripts/prepare_sd_current_vae_gate.py')}
 value['model_code_unchanged']=False;value['mathematical_kernel_objects_unchanged']=True;value['actual_thread_dispatch_compiler']=build_evidence;value['accepted_current_proof_sha256']=sha(accepted_path)
 value['configuration']=dict(vae_spatial_tile='mixed2048',generic_threads=8,vae_generic_threads=4,vae_blas_threads=4,thread_strategy='all4',im2col_mode='rows_256',group_norm='selective',softmax_scale='selective',cont_kernel='extended',pixel_kernel='ve',rgb_buffer='resident',png_encoder='ve',gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');value['scope']='current mixed2048 model with exact output-row dispatch gate extension to four or eight threads, VAE all-four-thread strategy; CLIP/UNet eight, VAE generic/BLAS four and idle remains four until next CLIP; 16 nodes/224 tiled calls; only backend orchestration object and archive rebuilt; 42 mathematical objects/archives unchanged; default unchanged; full graph incremental speedup unmeasured'
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Current VAE thread model tests audited:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])
if __name__=='__main__':main()
