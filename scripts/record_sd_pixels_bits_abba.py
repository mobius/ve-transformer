"""Recompute archived-build ABBA timers, dispatch, CPU errors and PNG bytes."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from benchmark_sd_pixels_bits_abba import verify_pair,PROOFS,LIBRARIES
from record_sd_pixels_bits_model import recompute_cpu_reference
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch
from record_sd_gemm_spatial_model import tile_dispatch
from record_sd_png_model import png_dispatch
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]

def guarded_csv(log):
 text=log.read_text()
 if 'Temperature guard stopped command' in text or text.count('Temperature summary:')!=1:raise RuntimeError('completed safe thermal guard required')
 paths=re.findall(r'log=(build/results/[^\s]+\.csv)',text)
 if len(paths)!=2:raise RuntimeError('thermal/fan CSVs required')
 rows=list(csv.DictReader((ROOT/paths[0]).open()))
 if not rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in rows):raise RuntimeError('unsafe thermal samples')
 return [ROOT/p for p in paths]

def audited_run(run,memory,guard,binary,checker,steps,cases):
 value=json.loads((run/'summary.json').read_text())
 fixed={'mode':'resident','tokenizer_mode':'resident','nlc_threads':'unified','nlc_mode':'openmp','vae_threads':8,'vae_blas_threads':4,'gelu_mode':'ve','binary_scalar_mode':'ve','im2col_mode':'rows_256','vae_spatial_tile':'8192','png_encoder':'ve','rgb_buffer':'resident','pixel_kernel':'ve'}
 if any(value.get(k)!=v for k,v in fixed.items()) or value['steps']!=steps or [r['reference_case'] for r in value['requests']]!=cases or any(value.get(k) for k in ('operator_profile_enabled','binary_shape_profile_enabled','image_output_profile_enabled')):raise RuntimeError('fixed optimized workload differs')
 checks=check_requests(run,value,binary,checker);rows_dispatch(run,value,'rows_256');tile_dispatch(run,value,'8192');png_dispatch(run,value,'ve')
 raw=(run/'native.log').read_text()
 if any(marker in raw for marker in ('SD_NLC_GEMM ','SD_IM2COL_SHAPE ','SD_IMAGE_OUTPUT_PROFILE ')):raise RuntimeError('unexpected detailed timing probes')
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S);image_times=[]
 for index,(body,request) in enumerate(zip(bodies,value['requests'])):
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
 entry=dict(artifacts=str(run.relative_to(ROOT)),memory_artifacts=str(memory.relative_to(ROOT)),request_seconds=[r['request_seconds'] for r in value['requests']],process_seconds=value['process_seconds'],image_output_seconds=image_times,cpu_checks=checks,cpu_recomputed_checks=recomputed,cpu_recomputed_png_checks=png_count,temperature=thermal(guard),fan_detail=fan_detail(guard),sampled_node_peak_gib=sampled['sampled_highest_used_kib']/1048576,sha256={str(p.relative_to(ROOT)):sha(p) for p in files},cpu_reference_sha256=reference_sha)
 return value,entry

def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();folder=a.benchmark.resolve();folder.relative_to(ROOT/'build');guard=a.guard_log.resolve();guard.relative_to(ROOT/'build');target=a.output.resolve();target.relative_to(ROOT/'docs/results')
 bench=json.loads((folder/'summary.json').read_text())
 if not bench['completed'] or bench['kind']!='sd_turbo_pixels_bits_dual_abba' or bench['same_binary'] is not False or bench['case_sequence']!='0,0,0,0' or [r['mode'] for r in bench['runs']]!=['baseline','candidate','candidate','baseline']:raise RuntimeError('complete dual-build ABBA required')
 if sha(folder/'benchmark_sd_pixels_bits_abba.py')!=bench['benchmark_sha256'] or sha(ROOT/'scripts/benchmark_sd_pixels_bits_abba.py')!=bench['benchmark_sha256']:raise RuntimeError('benchmark payload changed')
 arms,pair=verify_pair(ROOT/bench['builds']['baseline']['snapshot'],ROOT/bench['builds']['candidate']['snapshot'])
 if pair!=bench['pair'] or {n:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/n) for n in LIBRARIES}!=bench['library_sha256']:raise RuntimeError('source pair or libraries changed')
 files=[folder/'summary.json',folder/'benchmark_sd_pixels_bits_abba.py'];prior_runs=[];prior_checks=prior_png=0
 for name,arm in arms.items():
  if arm[2]!=bench['builds'][name]['manifest_sha256'] or arm[1]['sha256']['build/sd-baseline-ve/bin/sd']!=bench['builds'][name]['binary_sha256']:raise RuntimeError('arm build binding differs')
  prior_path=ROOT/PROOFS[name];prior=json.loads(prior_path.read_text());files.extend([prior_path,arm[0]/'build/sd-baseline-ve/manifest.json',arm[0]/'actual-pixels-object.o'])
  if prior['checker_sha256']!=bench['checker_sha256']:raise RuntimeError('same accepted checker required')
  for entry in prior['tests']:
   for path,digest in entry['sha256'].items():
    if sha(ROOT/path)!=digest:raise RuntimeError('prior model evidence changed')
   logs=[ROOT/n for n in entry['sha256'] if n.endswith('.log') and not n.endswith('native.log')]
   if len(logs)!=1:raise RuntimeError('prior guard required')
   steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[entry['name']]
   value,verified=audited_run(ROOT/entry['artifacts'],ROOT/entry['memory_artifacts'],logs[0],bench['builds'][name]['binary_sha256'],bench['checker_sha256'],steps,cases)
   verified.update(mode=name,name=entry['name']);prior_runs.append(verified);prior_checks+=verified['cpu_checks'];prior_png+=verified['cpu_recomputed_png_checks']
 runs=[];checks=png_count=0;baseline_outputs=None
 for index,run in enumerate(bench['runs']):
  name=run['mode'];value,entry=audited_run(ROOT/run['artifacts'],ROOT/run['memory_artifacts'],guard,bench['builds'][name]['binary_sha256'],bench['checker_sha256'],1,[0,0,0,0])
  if value['requests']!=run['requests'] or value['process_seconds']!=run['process_seconds'] or json.loads((ROOT/run['memory_artifacts']/'summary.json').read_text())!=run['memory'] or value['reference_artifacts']!=bench['reference_artifacts']:raise RuntimeError('formal aggregate differs from native evidence')
  outputs=[(q['trace_sha256'],q['png_sha256']) for q in value['requests']]
  if baseline_outputs is None:baseline_outputs=outputs
  if outputs!=baseline_outputs:raise RuntimeError('formal F32/PNG bytes differ')
  files.append(folder/('run%d.log'%index));entry['mode']=name;runs.append(entry);checks+=entry['cpu_checks'];png_count+=entry['cpu_recomputed_png_checks']
 if prior_checks!=124 or checks!=80 or prior_png!=20 or png_count!=16:raise RuntimeError('complete prior/formal CPU/PNG audits required')
 metrics={}
 for key,index in [('process',None)]+[('request'+str(i),i) for i in range(4)]:
  means={name:sum(r['process_seconds'] if index is None else r['request_seconds'][index] for r in runs if r['mode']==name)/2 for name in arms};metrics[key]={**means,'latency_reduction_percent':(1-means['candidate']/means['baseline'])*100}
 hot={name:dict(samples_seconds=[t for r in runs if r['mode']==name for t in r['request_seconds'][1:]]) for name in arms}
 for name in arms:hot[name]['mean_seconds']=sum(hot[name]['samples_seconds'])/6
 hot['latency_reduction_percent']=(1-hot['candidate']['mean_seconds']/hot['baseline']['mean_seconds'])*100
 if metrics!=bench['metrics'] or hot!=bench['hot_requests'] or not bench['all_trace_and_png_bytes_identical']:raise RuntimeError('recomputed formal metrics differ')
 image={name:sum(t for r in runs if r['mode']==name for t in r['image_output_seconds'][1:])/6 for name in arms}
 stage_comparison={}
 for stage in ('clip','unet','vae'):
  for part in ('backend_CPU','backend_BLAS','graph_compute','nlc_thread_prepare','nlc_thread_release'):
   stage_comparison[stage+'/'+part]={name:sum(sum(p['seconds'] for p in request['profile'] if p['stage']==stage and p['part']==part) for run in bench['runs'] if run['mode']==name for request in run['requests'][1:])/6 for name in arms}
 paired_arms=[]
 for before_index,after_index in ((0,1),(3,2)):
  before=sum(runs[before_index]['request_seconds'][1:])/3;after=sum(runs[after_index]['request_seconds'][1:])/3
  paired_arms.append(dict(baseline_arm=before_index,candidate_arm=after_index,baseline_seconds=before,candidate_seconds=after,latency_reduction_percent=(1-after/before)*100))
 output=dict(status='formal_dual_abba_verified',same_binary=False,full_graph_incremental_speedup_measured=True,pair=pair,builds=bench['builds'],metrics=metrics,hot_requests=hot,hot_image_output_mean_seconds=image,hot_stage_profile_seconds=stage_comparison,paired_hot_arm_means=paired_arms,runs=runs,prior_runs=prior_runs,prior_cpu_checks=prior_checks,formal_cpu_checks=checks,independent_cpu_checks=prior_checks+checks,cpu_recomputed_png_checks=prior_png+png_count,all_trace_and_png_bytes_identical=True,temperature=thermal(guard),fan_detail=fan_detail(guard),checker_sha256=bench['checker_sha256'],benchmark_sha256=bench['benchmark_sha256'],library_sha256=bench['library_sha256'],artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={n:sha(ROOT/n) for n in ('scripts/benchmark_sd_pixels_bits_abba.py','scripts/benchmark_sd_runtime.py','scripts/record_sd_pixels_bits_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_png_model.py')},scope='incremental old isolated O2 versus bit-pattern/reordered O2 pixels; two archived binaries with same common neural/main/PNG source and fixed runtime; one-step 512x512 case0 ABBA one cold/three hot per arm, six hot samples/mode; all prior/formal CPU errors recomputed and full PNG RGB bytes checked; link layout differs, cold caches uncontrolled, no arbitrary prompt/multistep performance claim')
 safe(output);target.write_text(json.dumps(output,indent=2)+'\n');print('Formal pixels bits dual ABBA audited:',target.relative_to(ROOT));print(hot)
if __name__=='__main__':main()
