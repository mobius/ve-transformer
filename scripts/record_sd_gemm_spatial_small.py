"""Audit guarded spatial GEMM tiling against current adopted tiles, full BLAS and FP64 samples."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from benchmark_nlc_gemm import input_value
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if os.environ.get("VE_TRANSFORMER_TEMPERATURE_SUPERVISED")!="1":raise RuntimeError("temperature supervision required")
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 log=a.log.resolve();log.relative_to(ROOT/'build');text=log.read_text()
 if 'Temperature summary:' not in text or 'Fan observation:' not in text:raise RuntimeError('completed guarded native process required')
 if text.count('TILE_COMPLETE shapes=6 candidates=3 arms=4 runs=4 threads=4')!=1 or 'Temperature guard stopped command' in text:raise RuntimeError('complete successful native run required')
 profile=a.profile.resolve();profile.relative_to(ROOT/'docs/results');v=json.loads(profile.read_text())
 configs=re.findall(r'TILE_CONFIG shape=(\d+) m=(\d+) n=(\d+) k=(\d+) lda=(\d+) ldb=(\d+) ldc=(\d+) actual_threads=(\d+) baseline_tile=(\d+)',text)
 expected=[(idx,m,n,k,k,k,n,4,0 if idx==0 else 8192) for idx,(m,n,k) in enumerate(((128,262144,1152),(256,262144,2304),(512,65536,4608),(256,65536,2304),(256,65536,4608),(128,262144,2304)))]
 if [tuple(map(int,r)) for r in configs]!=expected:raise RuntimeError('actual shape and threads differ')
 observed=v['requests'][1]['shapes_by_total_seconds']
 for _,m,n,k,lda,ldb,ldc,_,_ in expected:
  if not any(s['stage']=='vae' and tuple(s[x] for x in ('m','n','k','lda','ldb','ldc'))==(m,n,k,lda,ldb,ldc) for s in observed):raise RuntimeError('shape not observed in model')
 times=re.findall(r'TILE_TIME shape=(\d+) candidate=(\d+) arm=(\d+) tile=(\d+) rep=(\d+) seconds=([0-9.]+) checked=(\d+) max_error=([^\s]+)',text)
 samples=re.findall(r'TILE_SAMPLE shape=(\d+) candidate=(\d+) arm=(\d+) rep=(\d+) sample=(\d+) row=(\d+) column=(\d+) value=([^\s]+)',text)
 order=[(s,t,arm,t if arm in (1,2) else (0 if s==0 else 8192),r) for s in range(6) for t in (1024,2048,4096) for arm in range(4) for r in range(4)]
 if len(times)!=len(order) or len(samples)!=len(order)*20:raise RuntimeError('complete ordered timings and samples required')
 reference={};results=[]
 for s,m,n,k,lda,ldb,ldc,_,_ in expected:
  coords=[(0,0),(m-1,n-1),(0,n-1),(m-1,0)]+[((q*7919+17)%m,(q*104729+23)%n) for q in range(4,20)]
  if len(set(coords))!=20:raise RuntimeError('distinct samples required')
  reference[s]=(coords,[math.fsum(input_value(row*k+q,0x12345678)*input_value(col*k+q,0x9abcdef0) for q in range(k)) for row,col in coords])
 max_error=0.;records=[]
 for idx,(raw,key) in enumerate(zip(times,order)):
  s,t,arm,tile,rep=key
  if tuple(map(int,raw[:5]))!=key or int(raw[6])!=expected[s][1]*expected[s][2] or not math.isfinite(float(raw[5])) or float(raw[5])<=0 or not math.isfinite(float(raw[7])):raise RuntimeError('invalid timing/full comparison record')
  records.append(dict(shape=s,candidate=t,arm=arm,tile=tile,rep=rep,seconds=float(raw[5]),checked=int(raw[6]),max_full_difference=float(raw[7])))
  coords,ref=reference[s]
  for q,sm in enumerate(samples[idx*20:(idx+1)*20]):
   if tuple(map(int,sm[:5]))!=(s,t,arm,rep,q) or tuple(map(int,sm[5:7]))!=coords[q]:raise RuntimeError('sample ordering differs')
   val=float(sm[7]);err=abs(val-ref[q]);max_error=max(max_error,err)
   if not math.isfinite(val) or err>0.0001+0.0001*abs(ref[q]):raise RuntimeError('independent FP64 sample failed')
 for s,m,n,k,*_ in expected:
  if re.findall(r'TILE_INPUT shape='+str(s)+r' elements=(\d+) PASS',text)!=[str(m*k+n*k)]:raise RuntimeError('full unchanged input check required')
  for t in (1024,2048,4096):
   base=[r['seconds'] for r in records if r['shape']==s and r['candidate']==t and r['arm'] in (0,3) and r['rep']>0]
   candidate=[r['seconds'] for r in records if r['shape']==s and r['candidate']==t and r['arm'] in (1,2) and r['rep']>0]
   b=sum(base)/len(base);c=sum(candidate)/len(candidate)
   pairs=[]
   for ba,ca in ((0,1),(3,2)):
    bs=[r['seconds'] for r in records if r['shape']==s and r['candidate']==t and r['arm']==ba and r['rep']>0]
    cs=[r['seconds'] for r in records if r['shape']==s and r['candidate']==t and r['arm']==ca and r['rep']>0]
    bm=sum(bs)/len(bs);cm=sum(cs)/len(cs)
    pairs.append(dict(baseline_arm=ba,candidate_arm=ca,baseline_mean_seconds=bm,candidate_mean_seconds=cm,latency_reduction_percent=(bm-cm)/bm*100))
   results.append(dict(shape=s,m=m,n=n,k=k,tile=t,baseline_tile=0 if s==0 else 8192,paired_arm_means=pairs,both_pairs_positive=all(r['latency_reduction_percent']>0 for r in pairs),baseline_seconds=base,candidate_seconds=candidate,baseline_mean_seconds=b,candidate_mean_seconds=c,latency_reduction_percent=(b-c)/b*100))
 temp=ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)',text)[-1];fans=ROOT/re.findall(r'Fan observation:.*log=([^\s]+)',text)[-1]
 tr=list(csv.DictReader(temp.open()))
 if not tr or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in tr):raise RuntimeError('invalid thermal evidence')
 libraries={name:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/name) for name in ('libcblas.so','libblas_openmp.so')}
 run_matches=re.findall(r'Small tile artifacts: (build/results/[A-Za-z0-9_-]+)',text)
 if len(run_matches)!=1:raise RuntimeError('one frozen run payload required')
 run_dir=ROOT/run_matches[0]
 for source in ('tests/benchmark_sd_gemm_spatial_small.c','scripts/benchmark_sd_gemm_spatial_small.sh','build/sd-gemm-spatial-small/probe'):
  if sha(ROOT/source)!=sha(run_dir/Path(source).name):raise RuntimeError('run payload differs from audited source or binary')
 prior_path=ROOT/'docs/results/20261009T143151Z-sd-turbo-gemm-spatial-extended.json'
 prior=json.loads(prior_path.read_text())
 if libraries!=prior['nlc_library_sha256']:raise RuntimeError('NLC differs from accepted spatial probe')
 model_path=ROOT/'docs/results/20261009T222649Z-sd-turbo-group-norm-abba.json'
 model=json.loads(model_path.read_text())
 if model['status']!='formal_group_norm_abba_verified' or not model.get('cpu_validation_temperature') or model['independent_cpu_checks']!=142:raise RuntimeError('finalized current model required')
 manifest_path=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest_path)!=model['model_manifest_sha256']:raise RuntimeError('model manifest differs')
 manifest=json.loads(manifest_path.read_text())
 for name,digest in manifest['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('current model input changed')
 if sha(ROOT/'build/sd-baseline-ve/bin/sd')!=model['binary_sha256'] or sha(ROOT/'tests/check_sd_resident.py')!=model['checker_sha256']:raise RuntimeError('current model binary or checker changed')
 # Verify every path-bound SHA in current formal proof, including run source bindings.
 def verify_files(value):
  if isinstance(value,dict):
   for key,val in value.items():
    f=ROOT/key
    if isinstance(val,str) and re.fullmatch('[0-9a-f]{64}',val) and f.is_file() and sha(f)!=val:raise RuntimeError('model-bound artifact changed')
    verify_files(val)
  elif isinstance(value,list):
   for val in value:verify_files(val)
 verify_files(model)
 files=[log,temp,fans,profile,prior_path,model_path,manifest_path,ROOT/'scripts/benchmark_nlc_gemm.py',ROOT/'tests/benchmark_sd_gemm_spatial_small.c',ROOT/'scripts/benchmark_sd_gemm_spatial_small.sh',ROOT/'build/sd-gemm-spatial-small/probe']+list(run_dir.iterdir())
 output=dict(status='current_baseline_microbenchmark_verified',current_model_proof_sha256=sha(model_path),baseline_tiles=[0,8192,8192,8192,8192,8192],nlc_library_sha256=libraries,compiler_flags='-O2 -fno-fast-math -fno-associative-math -fopenmp',full_graph_speedup_measured=False,comparisons=results,records=records,full_blas_comparisons=len(records),full_compared_elements=sum(r['checked'] for r in records),independent_fp64_samples=len(samples),max_abs_sample_error=max_error,temperature=thermal(log),fan_detail=fan_detail(log),source_profile=str(profile.relative_to(ROOT)),source_profile_sha256=sha(profile),publisher_sha256=sha(Path(__file__)),artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},scope='six actual VAE shapes, fixed synthetic FP32 inputs, baseline tile 0 for rejected shape and 8192 for five adopted shapes, candidate tiles 1024/2048/4096, four threads, ABBA per tile with one warmup and three measurements per arm; complete baseline-BLAS comparisons plus twenty independent FP64 samples per call, not full CPU oracle; matrix-only timings; no model or whole-graph gain claim')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('Small spatial tiling microbenchmark audited:',target.relative_to(ROOT))
 for r in results:print('shape',r['shape'],'tile',r['tile'],'ms',round(r['baseline_mean_seconds']*1000,3),'->',round(r['candidate_mean_seconds']*1000,3),'reduction',round(r['latency_reduction_percent'],2))
if __name__=='__main__':main()
