"""Serial archived-build ABBA for the incremental O2 pixel-kernel change."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from benchmark_sd_runtime import sha,verify_snapshot
ROOT=Path(__file__).resolve().parents[1]
PROOFS={'baseline':'docs/results/20261009T172525Z-sd-turbo-pixels-bits-model.json','candidate':'docs/results/20261009T181647Z-sd-turbo-pixels-pack-model.json'}
LIBRARIES=('libcblas.so','libblas_sequential.so','libblas_openmp.so')

def verify_pair(baseline,candidate):
 arms={name:verify_snapshot(path,True) for name,path in (('baseline',baseline),('candidate',candidate))}
 proof_sha={}
 for name,(folder,value,digest) in arms.items():
  proof=ROOT/PROOFS[name];prior=json.loads(proof.read_text())
  if not prior['all_candidate_tests_completed'] or prior['completed_tests']!=['double','six','four'] or prior['independent_cpu_checks']!=62 or prior['model_manifest_sha256']!=digest or prior['binary_sha256']!=value['sha256']['build/sd-baseline-ve/bin/sd']:raise RuntimeError('snapshot differs from accepted model proof')
  if sha(folder/'actual-pixels-object.o')!=prior['actual_pixels_compiler']['pixels_object_sha256'] or prior['actual_pixels_compiler']['effective_pixels_optimization']!='-O2' or prior['actual_png_compiler']['main_optimization']!='-O0' or prior['actual_png_compiler']['effective_png_optimization']!='-O2':raise RuntimeError('accepted actual object/flags differ')
  if name=='candidate' and (not prior.get('actual_model_object_matches_independent_kernel') or not all(e.get('packing_input_domain_verified') for e in prior['tests'])):raise RuntimeError('candidate object and real packing inputs must be verified')
  proof_sha[name]=sha(proof)
 b,c=arms['baseline'][1],arms['candidate'][1];common=set(b['sha256'])&set(c['sha256'])
 changed={n for n in common if b['sha256'][n]!=c['sha256'][n]}
 extra={'src/ve_sd_turbo_pixels_pack.cpp'}
 if changed!={'cmake/nec-sd-baseline-overrides.cmake','build/sd-baseline-ve/bin/sd'} or set(b['sha256'])-set(c['sha256']) or set(c['sha256'])-set(b['sha256'])!=extra:raise RuntimeError('unexpected snapshot source differences')
 cmake='cmake/nec-sd-baseline-overrides.cmake'
 original=(arms['baseline'][0]/cmake).read_text();updated=(arms['candidate'][0]/cmake).read_text()
 if original.count('add_library(sd-pixels-ve STATIC "${project_root}/src/ve_sd_turbo_pixels_bits.cpp")')!=1 or original.replace('add_library(sd-pixels-ve STATIC "${project_root}/src/ve_sd_turbo_pixels_bits.cpp")','add_library(sd-pixels-ve STATIC "${project_root}/src/ve_sd_turbo_pixels_pack.cpp")')!=updated:raise RuntimeError('CMake change exceeds isolated pixel source')
 for key in ('framework_revision','ggml_revision','compiler_standard','precision','compiler_optimization','nlc_mode','dynamic_dependencies','generic_thread_runtime','nlc_thread_lifecycle'):
  if b[key]!=c[key]:raise RuntimeError('build configuration differs')
 return arms,dict(changed_common_artifacts=sorted(changed),candidate_only_artifacts=sorted(extra),matching_common_artifacts=len(common)-len(changed),accepted_proof_sha256=proof_sha,same_binary=False)

def main():
 if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':raise RuntimeError('temperature supervision required')
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args();reference=a.reference.resolve();reference.relative_to(ROOT/'build')
 arms,pair=verify_pair(a.baseline,a.candidate);checker=ROOT/'tests/check_sd_resident.py';library_sha={n:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/n) for n in LIBRARIES};checker_sha=sha(checker)
 if any(json.loads((ROOT/PROOFS[name]).read_text())['checker_sha256']!=checker_sha for name in arms):raise RuntimeError('same accepted checker required')
 folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-pixels-pack-abba',time.gmtime());folder.mkdir()
 report=dict(completed=False,kind='sd_turbo_pixels_pack_dual_abba',same_binary=False,pair=pair,case_sequence='0,0,0,0',checker_sha256=checker_sha,benchmark_sha256=sha(Path(__file__)),reference_artifacts=str(reference.relative_to(ROOT)),library_sha256=library_sha,builds={name:dict(snapshot=str(arm[0].relative_to(ROOT)),manifest_sha256=arm[2],binary_sha256=arm[1]['sha256']['build/sd-baseline-ve/bin/sd']) for name,arm in arms.items()},runs=[])
 def save():(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 save();(folder/'benchmark_sd_pixels_pack_abba.py').write_bytes(Path(__file__).read_bytes())
 for index,name in enumerate(('baseline','candidate','candidate','baseline')):
  current,current_pair=verify_pair(a.baseline,a.candidate)
  if current_pair!=pair or sha(checker)!=checker_sha or {n:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/n) for n in LIBRARIES}!=library_sha:raise RuntimeError('benchmark inputs changed')
  print('Pixels pack dual ABBA',index,name,flush=True)
  command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/sample_ve_memory.py'),'--timeout','600','--',str(ROOT/'.venv/bin/python'),str(checker),'--build-snapshot',str(current[name][0]),'--reference',str(reference),'--mode','resident','--tokenizer','resident','--nlc-threads','unified','--binary-scalar','ve','--vae-blas-threads','4','--gelu','ve','--cases','0,0,0,0','--im2col-mode','rows_256','--vae-spatial-tile','8192','--png-encoder','ve','--rgb-buffer','resident','--pixel-kernel','ve']
  path=folder/('run%d.log'%index)
  with path.open('w') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  text=path.read_text();native=re.findall(r'Native sequential requests verified: (build/results/[^\s]+)',text);memory=re.findall(r'VE memory samples: (build/results/[^\s]+)',text)
  if r.returncode or len(native)!=1 or len(memory)!=1:raise RuntimeError('complete native/memory arm required')
  value=json.loads((ROOT/native[0]/'summary.json').read_text());sampled=json.loads((ROOT/memory[0]/'summary.json').read_text())
  if not value['completed'] or not sampled['completed'] or sampled['final_used_kib']!=131072 or value['binary_sha256']!=report['builds'][name]['binary_sha256'] or value['checker_sha256']!=checker_sha or value['steps']!=1 or [q['reference_case'] for q in value['requests']]!=[0,0,0,0] or value['pixel_kernel']!='ve' or not all(len(q['checks'])==5 and all(c['passed'] for c in q['checks']) for q in value['requests']):raise RuntimeError('verified matching arm required')
  report['runs'].append(dict(mode=name,artifacts=native[0],memory_artifacts=memory[0],requests=value['requests'],process_seconds=value['process_seconds'],memory=sampled));save()
 equality=all([(q['trace_sha256'],q['png_sha256']) for q in r['requests']]==[(q['trace_sha256'],q['png_sha256']) for q in report['runs'][0]['requests']] for r in report['runs'])
 if not equality:raise RuntimeError('all F32/PNG bytes must match')
 metrics={}
 for key,index in [('process',None)]+[('request'+str(i),i) for i in range(4)]:
  mean={name:sum(r['process_seconds'] if index is None else r['requests'][index]['request_seconds'] for r in report['runs'] if r['mode']==name)/2 for name in arms};metrics[key]={**mean,'latency_reduction_percent':(1-mean['candidate']/mean['baseline'])*100}
 hot={name:[q['request_seconds'] for r in report['runs'] if r['mode']==name for q in r['requests'][1:]] for name in arms};means={name:sum(values)/len(values) for name,values in hot.items()}
 report.update(completed=True,metrics=metrics,hot_requests={**{name:dict(samples_seconds=hot[name],mean_seconds=means[name]) for name in arms},'latency_reduction_percent':(1-means['candidate']/means['baseline'])*100},all_trace_and_png_bytes_identical=True,scope='two archived O2 pixel builds, fixed one-step 512x512 prompt, ABBA one cold/three hot requests per arm; same neural/main/PNG source, changed pixel target and executable; link layout differs, cold caches uncontrolled; incremental full-model comparison')
 save();print('Pixels pack dual ABBA completed:',folder.relative_to(ROOT),flush=True)
if __name__=='__main__':main()
