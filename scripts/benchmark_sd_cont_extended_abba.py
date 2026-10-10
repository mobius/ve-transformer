"""Serial same-binary CONT-only ABBA with fixed accepted model configuration."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from benchmark_sd_runtime import sha
from temperature_guard import discover,sample
ROOT=Path(__file__).resolve().parents[1]
def main():
 if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':raise RuntimeError('temperature supervision required')
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args();reference=a.reference.resolve();reference.relative_to(ROOT/'build');proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');accepted=json.loads(proof.read_text());manifest_path=ROOT/'build/sd-baseline-ve/manifest.json';manifest=json.loads(manifest_path.read_text());checker=ROOT/'tests/check_sd_resident.py';binary=ROOT/'build/sd-baseline-ve/bin/sd'
 if not accepted['all_candidate_tests_completed'] or accepted['independent_cpu_checks']!=62 or accepted['cpu_recomputed_png_checks']!=10 or not accepted['actual_model_object_matches_independent_kernel'] or sha(manifest_path)!=accepted['model_manifest_sha256'] or sha(checker)!=accepted['checker_sha256'] or sha(binary)!=accepted['binary_sha256']:raise RuntimeError('fully accepted matching CONT model required')
 folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-cont-extended-abba',time.gmtime());folder.mkdir();report=dict(completed=False,kind='sd_turbo_cont_extended_same_binary_abba',same_binary=True,case_sequence='0,0,0,0',proof=str(proof.relative_to(ROOT)),proof_sha256=sha(proof),model_manifest_sha256=sha(manifest_path),binary_sha256=sha(binary),checker_sha256=sha(checker),benchmark_sha256=sha(Path(__file__)),reference_artifacts=str(reference.relative_to(ROOT)),runs=[])
 def save():(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 save();(folder/'benchmark_sd_cont_extended_abba.py').write_bytes(Path(__file__).read_bytes())
 channels=discover()
 for index,mode in enumerate(('original','extended','extended','original')):
  # Cool only between measured processes; never throttle a timed request.
  cooling_started=time.monotonic();stable_since=None
  while True:
   readings=sample(channels);cpu=max(v for k,v in readings.items() if k.startswith('cpu/'));ve=max(v for k,v in readings.items() if not k.startswith('cpu/'))
   if cpu<=65 and ve<=55:
    if stable_since is None:stable_since=time.monotonic()
    if time.monotonic()-stable_since>=10:break
   else:stable_since=None
   if time.monotonic()-cooling_started>300:raise RuntimeError('safe pre-arm cooling not reached')
   time.sleep(1)
  report.setdefault('pre_arm_cooling',[]).append(dict(arm=index,seconds=time.monotonic()-cooling_started,cpu_c=cpu,ve_c=ve));save();print('Extended CONT pre-arm cooling completed',index,flush=True)
  if sha(proof)!=report['proof_sha256'] or sha(manifest_path)!=report['model_manifest_sha256'] or sha(binary)!=report['binary_sha256'] or sha(checker)!=report['checker_sha256']:raise RuntimeError('same model/proof/checker changed')
  for name,digest in manifest['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('model source changed')
  command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/sample_ve_memory.py'),'--timeout','600','--',str(ROOT/'.venv/bin/python'),str(checker),'--reference',str(reference),'--mode','resident','--tokenizer','resident','--nlc-threads','unified','--binary-scalar','ve','--vae-blas-threads','4','--gelu','ve','--cases','0,0,0,0','--im2col-mode','rows_256','--vae-spatial-tile','8192','--png-encoder','ve','--rgb-buffer','resident','--pixel-kernel','ve']
  env=dict(os.environ,SD_VE_CONT='extended' if mode=='extended' else '1',SD_CONT_SHAPE_PROFILE='0');path=folder/('run%d.log'%index);print('Extended CONT same-binary ABBA',index,mode,flush=True)
  with path.open('w') as log:r=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  text=path.read_text();native=re.findall(r'Native sequential requests verified: (build/results/[^\s]+)',text);memory=re.findall(r'VE memory samples: (build/results/[^\s]+)',text)
  if r.returncode or len(native)!=1 or len(memory)!=1:raise RuntimeError('completed native/memory arm required')
  value=json.loads((ROOT/native[0]/'summary.json').read_text());sampled=json.loads((ROOT/memory[0]/'summary.json').read_text());raw=(ROOT/native[0]/'native.log').read_text()
  expected=[('clip',str(23 if mode=='extended' else 0),str(69 if mode=='extended' else 92),'1'),('unet',str(45 if mode=='extended' else 15),str(213 if mode=='extended' else 243),'1'),('vae',str(3 if mode=='extended' else 0),str(40 if mode=='extended' else 43),'1')]*4
  observed=re.findall(r'SD_CONT_DISPATCH stage=(clip|unet|vae) optimized=(\d+) fallback=(\d+) enabled=(0|1)',raw)
  if observed!=expected or 'SD_CONT_SHAPE ' in raw or len(observed)!=raw.count('SD_CONT_DISPATCH '):raise RuntimeError('actual CONT-only switch differs')
  if not value['completed'] or not sampled['completed'] or sampled['final_used_kib']!=131072 or value['binary_sha256']!=report['binary_sha256'] or value['checker_sha256']!=report['checker_sha256'] or value['steps']!=1 or [q['reference_case'] for q in value['requests']]!=[0,0,0,0] or value['pixel_kernel']!='ve' or any(value.get(k) for k in ('operator_profile_enabled','binary_shape_profile_enabled','image_output_profile_enabled')) or not all(len(q['checks'])==5 and all(c['passed'] for c in q['checks']) for q in value['requests']):raise RuntimeError('fixed accepted unprofiled workload required')
  report['runs'].append(dict(mode=mode,artifacts=native[0],memory_artifacts=memory[0],requests=value['requests'],process_seconds=value['process_seconds'],memory=sampled));save()
 identical=all([(q['trace_sha256'],q['png_sha256']) for q in r['requests']]==[(q['trace_sha256'],q['png_sha256']) for q in report['runs'][0]['requests']] for r in report['runs'])
 if not identical:raise RuntimeError('all F32/PNG bytes must match')
 metrics={}
 for key,index in [('process',None)]+[('request'+str(i),i) for i in range(4)]:
  means={mode:sum(r['process_seconds'] if index is None else r['requests'][index]['request_seconds'] for r in report['runs'] if r['mode']==mode)/2 for mode in ('original','extended')};metrics[key]={**means,'latency_reduction_percent':(1-means['extended']/means['original'])*100}
 hot={mode:dict(samples_seconds=[q['request_seconds'] for r in report['runs'] if r['mode']==mode for q in r['requests'][1:]]) for mode in ('original','extended')}
 for mode in hot:hot[mode]['mean_seconds']=sum(hot[mode]['samples_seconds'])/6
 hot['latency_reduction_percent']=(1-hot['extended']['mean_seconds']/hot['original']['mean_seconds'])*100
 report.update(completed=True,metrics=metrics,hot_requests=hot,all_trace_and_png_bytes_identical=True,scope='same binary/checker, only SD_VE_CONT switches original three layouts versus extended twelve layouts; fixed single-VE one-step 512x512 case0, ABBA one cold/three hot per arm, six hot samples/mode; no detailed probes; cold loading cache uncontrolled, no general prompt/multistep claim');save();print('Extended CONT same-binary ABBA completed:',folder.relative_to(ROOT),flush=True)
if __name__=='__main__':main()
