"""Independently audit same-binary CONT ABBA and all prior model CPU/PNG outputs."""
import argparse
import json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import audited_run
from record_sd_cont_extended_model import cont_dispatch,compiler_evidence
from record_sd_png_model import png_compiler_evidence
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();folder=a.benchmark.resolve();folder.relative_to(ROOT/'build');guard=a.guard_log.resolve();guard.relative_to(ROOT/'build');proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');target=a.output.resolve();target.relative_to(ROOT/'docs/results');bench=json.loads((folder/'summary.json').read_text());prior=json.loads(proof.read_text())
 if not bench['completed'] or bench['kind']!='sd_turbo_cont_extended_same_binary_abba' or bench['same_binary'] is not True or bench['case_sequence']!='0,0,0,0' or [r['mode'] for r in bench['runs']]!=['original','extended','extended','original'] or bench['proof']!=str(proof.relative_to(ROOT)) or bench['proof_sha256']!=sha(proof):raise RuntimeError('complete same-binary CONT ABBA required')
 cooling=bench.get('pre_arm_cooling',[])
 if len(cooling)!=4 or [x['arm'] for x in cooling]!=[0,1,2,3] or any(x['seconds']<10 or x['seconds']>301 or not 0<x['cpu_c']<=65 or not 0<x['ve_c']<=55 for x in cooling):raise RuntimeError('all four pre-arm cooling gates required')
 if not prior['all_candidate_tests_completed'] or prior['independent_cpu_checks']!=62 or prior['cpu_recomputed_png_checks']!=10 or bench['model_manifest_sha256']!=prior['model_manifest_sha256'] or bench['binary_sha256']!=prior['binary_sha256'] or bench['checker_sha256']!=prior['checker_sha256']:raise RuntimeError('matching fully verified model required')
 if sha(folder/'benchmark_sd_cont_extended_abba.py')!=bench['benchmark_sha256'] or sha(ROOT/'scripts/benchmark_sd_cont_extended_abba.py')!=bench['benchmark_sha256']:raise RuntimeError('benchmark payload changed')
 compiler=compiler_evidence();png=png_compiler_evidence()
 if compiler!=prior['actual_cont_compiler'] or png!=prior['actual_png_compiler']:raise RuntimeError('actual model objects/flags changed')
 current=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(current)!=bench['model_manifest_sha256']:raise RuntimeError('model manifest changed')
 for name,digest in json.loads(current.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model build input changed')
 files=[folder/'summary.json',folder/'benchmark_sd_cont_extended_abba.py',proof,current];prior_runs=[];prior_checks=prior_png=0
 for entry in prior['tests']:
  for name,digest in entry['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('prior model evidence changed')
  logs=[ROOT/n for n in entry['sha256'] if n.endswith('.log') and not n.endswith('native.log')]
  if len(logs)!=1:raise RuntimeError('prior guard required')
  steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[entry['name']]
  value,verified=audited_run(ROOT/entry['artifacts'],ROOT/entry['memory_artifacts'],logs[0],bench['binary_sha256'],bench['checker_sha256'],steps,cases);dispatch=cont_dispatch(ROOT/entry['artifacts'],value,'extended')
  if dispatch!=entry['cont_dispatch']:raise RuntimeError('prior candidate dispatch changed')
  verified.update(name=entry['name'],mode='extended',cont_dispatch=dispatch);prior_runs.append(verified);prior_checks+=verified['cpu_checks'];prior_png+=verified['cpu_recomputed_png_checks']
 runs=[];checks=png_count=0;baseline=None
 for index,run in enumerate(bench['runs']):
  native,entry=audited_run(ROOT/run['artifacts'],ROOT/run['memory_artifacts'],guard,bench['binary_sha256'],bench['checker_sha256'],1,[0,0,0,0]);dispatch=cont_dispatch(ROOT/run['artifacts'],native,'ve' if run['mode']=='original' else 'extended')
  if native['requests']!=run['requests'] or native['process_seconds']!=run['process_seconds'] or json.loads((ROOT/run['memory_artifacts']/'summary.json').read_text())!=run['memory'] or native['reference_artifacts']!=bench['reference_artifacts']:raise RuntimeError('formal aggregate differs from raw evidence')
  outputs=[(q['trace_sha256'],q['png_sha256']) for q in native['requests']]
  if baseline is None:baseline=outputs
  if outputs!=baseline:raise RuntimeError('formal F32/PNG bytes differ')
  files.append(folder/('run%d.log'%index));entry.update(mode=run['mode'],cont_dispatch=dispatch);runs.append(entry);checks+=entry['cpu_checks'];png_count+=entry['cpu_recomputed_png_checks']
 if prior_checks!=62 or checks!=80 or prior_png!=10 or png_count!=16:raise RuntimeError('142 CPU/26 PNG recomputed checks required')
 metrics={}
 for key,index in [('process',None)]+[('request'+str(i),i) for i in range(4)]:
  means={mode:sum(r['process_seconds'] if index is None else r['request_seconds'][index] for r in runs if r['mode']==mode)/2 for mode in ('original','extended')};metrics[key]={**means,'latency_reduction_percent':(1-means['extended']/means['original'])*100}
 hot={mode:dict(samples_seconds=[t for r in runs if r['mode']==mode for t in r['request_seconds'][1:]]) for mode in ('original','extended')}
 for mode in hot:hot[mode]['mean_seconds']=sum(hot[mode]['samples_seconds'])/6
 hot['latency_reduction_percent']=(1-hot['extended']['mean_seconds']/hot['original']['mean_seconds'])*100
 if metrics!=bench['metrics'] or hot!=bench['hot_requests'] or not bench['all_trace_and_png_bytes_identical']:raise RuntimeError('recomputed formal metrics differ')
 stages={}
 for stage in ('clip','unet','vae'):
  for part in ('backend_CPU','backend_BLAS','graph_compute','nlc_thread_prepare','nlc_thread_release'):
   stages[stage+'/'+part]={mode:sum(sum(p['seconds'] for p in request['profile'] if p['stage']==stage and p['part']==part) for run in bench['runs'] if run['mode']==mode for request in run['requests'][1:])/6 for mode in ('original','extended')}
 paired=[]
 for b,c in ((0,1),(3,2)):
  before=sum(runs[b]['request_seconds'][1:])/3;after=sum(runs[c]['request_seconds'][1:])/3;paired.append(dict(baseline_arm=b,candidate_arm=c,baseline_seconds=before,candidate_seconds=after,latency_reduction_percent=(1-after/before)*100))
 image={mode:sum(t for r in runs if r['mode']==mode for t in r['image_output_seconds'][1:])/6 for mode in ('original','extended')}
 report=dict(pre_arm_cooling=cooling,status='formal_cont_extended_abba_verified',same_binary=True,full_graph_incremental_speedup_measured=True,model_manifest_sha256=bench['model_manifest_sha256'],binary_sha256=bench['binary_sha256'],checker_sha256=bench['checker_sha256'],benchmark_sha256=bench['benchmark_sha256'],actual_cont_compiler=compiler,actual_png_compiler=png,metrics=metrics,hot_requests=hot,hot_stage_profile_seconds=stages,hot_image_output_mean_seconds=image,paired_hot_arm_means=paired,runs=runs,prior_runs=prior_runs,prior_cpu_checks=prior_checks,formal_cpu_checks=checks,independent_cpu_checks=prior_checks+checks,cpu_recomputed_png_checks=prior_png+png_count,all_trace_and_png_bytes_identical=True,temperature=thermal(guard),fan_detail=fan_detail(guard),artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={n:sha(ROOT/n) for n in ('scripts/benchmark_sd_cont_extended_abba.py','scripts/record_sd_cont_extended_model.py','scripts/record_sd_pixels_pack_abba.py','scripts/record_sd_pixels_pack_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_png_model.py')},scope='same binary/checker, only SD_VE_CONT switches original three UNet layouts versus extended twelve CLIP/UNet/VAE layouts, fixed one-step 512x512 case0 ABBA one cold/three hot per arm, six hot samples/mode; all prior/formal CPU/PNG recomputed; cold caches uncontrolled, no arbitrary prompt/multistep performance claim')
 safe(report);target.write_text(json.dumps(report,indent=2)+'\n');print('Extended CONT formal ABBA audited:',target.relative_to(ROOT));print(hot)
if __name__=='__main__':main()
