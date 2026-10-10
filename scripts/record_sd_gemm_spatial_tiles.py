"""Audit guarded spatial GEMM tiling against full BLAS and FP64 samples."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from benchmark_nlc_gemm import input_value
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 log=a.log.resolve();log.relative_to(ROOT/'build');text=log.read_text()
 if text.count('TILE_COMPLETE shapes=2 candidates=4 arms=4 runs=4 threads=4')!=1 or 'Temperature guard stopped command' in text:raise RuntimeError('complete successful native run required')
 profile=a.profile.resolve();profile.relative_to(ROOT/'docs/results');v=json.loads(profile.read_text())
 configs=re.findall(r'TILE_CONFIG shape=(\d+) m=(\d+) n=(\d+) k=(\d+) lda=(\d+) ldb=(\d+) ldc=(\d+) actual_threads=(\d+)',text)
 expected=[(0,128,262144,1152,1152,1152,262144,4),(1,256,262144,2304,2304,2304,262144,4)]
 if [tuple(map(int,r)) for r in configs]!=expected:raise RuntimeError('actual shape and threads differ')
 observed=v['requests'][1]['shapes_by_total_seconds']
 for _,m,n,k,lda,ldb,ldc,_ in expected:
  if not any(s['stage']=='vae' and tuple(s[x] for x in ('m','n','k','lda','ldb','ldc'))==(m,n,k,lda,ldb,ldc) for s in observed):raise RuntimeError('shape not observed in model')
 times=re.findall(r'TILE_TIME shape=(\d+) candidate=(\d+) arm=(\d+) tile=(\d+) rep=(\d+) seconds=([0-9.]+) checked=(\d+) max_error=([^\s]+)',text)
 samples=re.findall(r'TILE_SAMPLE shape=(\d+) candidate=(\d+) arm=(\d+) rep=(\d+) sample=(\d+) row=(\d+) column=(\d+) value=([^\s]+)',text)
 order=[(s,t,arm,t if arm in (1,2) else 0,r) for s in range(2) for t in (8192,16384,32768,65536) for arm in range(4) for r in range(4)]
 if len(times)!=len(order) or len(samples)!=len(order)*20:raise RuntimeError('complete ordered timings and samples required')
 reference={};results=[]
 for s,m,n,k,lda,ldb,ldc,_ in expected:
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
  for t in (8192,16384,32768,65536):
   base=[r['seconds'] for r in records if r['shape']==s and r['candidate']==t and r['arm'] in (0,3) and r['rep']>0]
   candidate=[r['seconds'] for r in records if r['shape']==s and r['candidate']==t and r['arm'] in (1,2) and r['rep']>0]
   b=sum(base)/len(base);c=sum(candidate)/len(candidate)
   results.append(dict(shape=s,m=m,n=n,k=k,tile=t,baseline_seconds=base,candidate_seconds=candidate,baseline_mean_seconds=b,candidate_mean_seconds=c,latency_reduction_percent=(b-c)/b*100))
 temp=ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)',text)[-1];fans=ROOT/re.findall(r'Fan observation:.*log=([^\s]+)',text)[-1]
 tr=list(csv.DictReader(temp.open()))
 if not tr or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in tr):raise RuntimeError('invalid thermal evidence')
 files=[log,temp,fans,profile,ROOT/'tests/benchmark_sd_gemm_spatial_tiles.c',ROOT/'scripts/benchmark_sd_gemm_spatial_tiles.sh',ROOT/'build/sd-gemm-spatial-tiles/probe']
 output=dict(status='microbenchmark_verified',full_graph_speedup_measured=False,comparisons=results,records=records,full_blas_comparisons=len(records),full_compared_elements=sum(r['checked'] for r in records),independent_fp64_samples=len(samples),max_abs_sample_error=max_error,temperature=thermal(log),fan_detail=fan_detail(log),source_profile=str(profile.relative_to(ROOT)),source_profile_sha256=sha(profile),publisher_sha256=sha(Path(__file__)),artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},scope='two actual VAE shapes, fixed synthetic FP32 inputs, four threads, ABBA per tile with one warmup and three measurements per arm; complete baseline-BLAS comparisons plus twenty independent FP64 samples per call, not full CPU oracle; matrix-only timings; no model or whole-graph gain claim')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('Spatial tiling microbenchmark audited:',target.relative_to(ROOT))
 for r in results:print('shape',r['shape'],'tile',r['tile'],'ms',round(r['baseline_mean_seconds']*1000,3),'->',round(r['candidate_mean_seconds']*1000,3),'reduction',round(r['latency_reduction_percent'],2))
if __name__=='__main__':main()
