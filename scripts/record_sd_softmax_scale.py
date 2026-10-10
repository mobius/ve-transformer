"""Independent float32 audit of standalone copy/scale row primitives."""
import argparse,json,re,math
from pathlib import Path
import numpy as np
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');out=a.output.resolve();out.relative_to(ROOT/'docs/results');text=log.read_text();paths=guarded_csv(log)
 if 'SOFTMAX_SCALE_BATCH_PASS fixtures=72 output_pairs=144 timings=240' not in text:raise RuntimeError('complete hardware batch required')
 match=re.search(r'^Softmax scale artifacts: (build/results/[^\s]+)$',text,re.M)
 if not match:raise RuntimeError('artifact folder absent')
 folder=(ROOT/match[1]).resolve();folder.relative_to(ROOT/'build/results');checks=re.findall(r'CHECK id=(\d+) n=(\d+) scale=([^ ]+) both_ops_bitwise_canary=PASS',text)
 scales=[np.float32(x) for x in (.000244140625,.00031234567,.001953125,.1,.33333334,.5,1.,2.)];expected=[];files=[];case=0;ieee_differences=0;affected_outputs=0
 edges=[0,0x80000000,1,0x80000001,0x00800000,0x80800000,0x7f800000,0xff800000,0x7f7fffff,0xff7fffff]
 for n in (1,7,64,77,255,256,257,1024,4096):
  state=0x7391ab;bits=[]
  for i in range(n):
   state=(state*1664525+1013904223)&0xffffffff;bits.append(edges[i%31] if i%31<10 else 0x3f000000+(state&0x7fffff))
  source=np.array(bits,dtype='<u4')
  for scale in scales:
   expected.append((case,n,int(scale.view('uint32'))));inputs=folder/f'case{case}-input.f32';files.append(inputs)
   if not np.array_equal(np.fromfile(inputs,dtype='<u4'),source):raise RuntimeError('input reconstruction differs')
   with np.errstate(over='ignore',under='ignore',invalid='ignore'):reference=np.multiply(source.view('<f4'),scale,dtype=np.float32).view('<u4')
   for op in ('copy','scale'):
    for mode in ('baseline','candidate'):
     f=folder/f'case{case}-{op}-{mode}.f32';files.append(f)
     actual=np.fromfile(f,dtype='<u4')
     flushed=reference.copy();subnormal=((flushed&0x7fffffff)>0)&((flushed&0x7fffffff)<0x00800000);flushed[subnormal]&=0x80000000
     if not np.array_equal(actual,flushed):
      bad=np.flatnonzero(actual!=flushed)[:10];raise RuntimeError('output differs beyond signed subnormal flushing: '+str((case,n,float(scale),op,mode,[(int(i),hex(int(source[i])),hex(int(actual[i])),hex(int(flushed[i]))) for i in bad])))
     differences=int(np.count_nonzero(actual!=reference));ieee_differences+=differences;affected_outputs+=int(differences>0)
   case+=1
 if [(int(i),int(n),int(np.float32(s).view('uint32'))) for i,n,s in checks]!=expected:raise RuntimeError('fixture order or float32 scale differs')
 if set(folder.glob('*.f32'))!=set(files):raise RuntimeError('unexpected raw output set')
 rows=re.findall(r'TIME n=(\d+) op=(copy|scale) rep=(\d+) arm=(\d+) mode=(baseline|candidate) calls=5000 seconds=([0-9.]+)',text)
 order=[(str(n),op,str(rep),str(arm),'candidate' if arm in (1,2) else 'baseline') for n in (64,77,256,1024,4096) for op in ('copy','scale') for rep in range(6) for arm in range(4)]
 if [r[:5] for r in rows]!=order or any(not math.isfinite(float(r[5])) or float(r[5])<=0 for r in rows):raise RuntimeError('ordered positive timings required')
 results=[]
 for n in (64,77,256,1024,4096):
  for op in ('copy','scale'):
   selected=[r for r in rows if int(r[0])==n and r[1]==op];samples={m:[float(r[5])/5000 for r in selected if r[4]==m] for m in ('baseline','candidate')};means={m:sum(x)/len(x) for m,x in samples.items()};pairs=[]
   for left,right in ((0,1),(3,2)):
    b=sum(float(r[5]) for r in selected if int(r[3])==left)/30000;c=sum(float(r[5]) for r in selected if int(r[3])==right)/30000;pairs.append((1-c/b)*100)
   results.append(dict(columns=n,operation=op,baseline_seconds=means['baseline'],candidate_seconds=means['candidate'],latency_reduction_percent=(1-means['candidate']/means['baseline'])*100,paired_reduction_percent=pairs,samples_seconds=samples))
 for line in (3,10):
  if f'src/ve_sd_turbo_softmax_scale.c, line {line}: Vectorized loop.' not in text:raise RuntimeError('candidate vector compiler evidence absent')
 archive=ROOT/'build/accepted/softmax-shapes-20261009T201945Z';index=json.loads((archive/'archive-sha256.json').read_text());manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=index['build/sd-baseline-ve/manifest.json']:raise RuntimeError('accepted model manifest changed')
 for name,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=h:raise RuntimeError('model input changed')
 pilot=ROOT/'build/diagnostics/softmax-scale-pilot-20261009T203613Z';pilot_index=json.loads((pilot/'archive-sha256.json').read_text())
 for name,h in pilot_index.items():
  if sha(pilot/name)!=h:raise RuntimeError('pilot evidence changed')
 names=('src/ve_sd_turbo_softmax_scale.c','tests/benchmark_sd_softmax_scale.cpp','scripts/benchmark_sd_softmax_scale.sh','build/sd-softmax-scale-probe/baseline.o','build/sd-softmax-scale-probe/candidate.o','build/sd-softmax-scale-probe/probe','build/sd-baseline-ve/manifest.json','build/sd-baseline-overlay/sd-ggml-cpu.c','docs/results/20261009T201945Z-sd-turbo-softmax-shapes.json','build/diagnostics/softmax-scale-pilot-20261009T203613Z/archive-sha256.json');files += [ROOT/n for n in names]+[log]+paths
 v=dict(status='standalone_softmax_scale_ftz_verified',results=results,fixtures=72,output_pairs=144,ieee_strict_reference_pass=False,ieee_different_elements=ieee_differences,ieee_affected_outputs=affected_outputs,reference_policy='CPU float32 product followed by signed flushing of nonzero subnormal results; strict IEEE reference fails',independent_cpu_output_checks=288,timed_batches=240,calls_per_batch=5000,model_unchanged=True,pilot_files=len(pilot_index),temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(x.relative_to(ROOT)):sha(x) for x in files},publisher_sha256=sha(Path(__file__)),scope='single worker hot row primitives, synthetic signed float32 edges, eight positive scales; independently reproduced O1 scalar algorithm versus strict O2 candidate, not actual ggml object or model speedup; timing copy scale=0.5 and in-place scale=1')
 safe(v);out.write_text(json.dumps(v,indent=2)+'\n');print('Softmax scale CPU audit passed:',out.relative_to(ROOT));print([(r['columns'],r['operation'],r['baseline_seconds']*1e6,r['candidate_seconds']*1e6) for r in results])
if __name__=='__main__':main()
