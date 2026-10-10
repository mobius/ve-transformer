"""Independent sequential FP64 reference and guarded batch timing audit."""
import argparse,json,re,struct,math
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');out=a.output.resolve();out.relative_to(ROOT/'docs/results');text=log.read_text()
 if 'SOFTMAX_SUM_BATCH_PASS checks=108 timings=120' not in text:raise RuntimeError('complete VE batch test required')
 paths=guarded_csv(log);checks=re.findall(r'CHECK n=(\d+) fixture=(\d+) baseline_bits=(\d+) candidate_bits=(\d+) PASS',text);expected=[]
 for n in (1,7,64,77,255,256,257,1024,4096):
  state=0x91ab73
  for fixture in range(12):
   total=0.0
   for i in range(n):
    state=(state*1664525+1013904223)&0xffffffff;b=0x3f000000+(state&0x7fffff)
    if i%31==0:b=0
    if i%31==1:b=1
    if i%31==2:b=0x00800000
    if i%31==3:b=0x3f800000
    total+=struct.unpack('<f',struct.pack('<I',b))[0]
   bits=struct.unpack('<Q',struct.pack('<d',total))[0];expected.append((str(n),str(fixture),str(bits),str(bits)))
 if checks!=expected:raise RuntimeError('108 independently recomputed sequential sums differ')
 rows=re.findall(r'TIME n=(\d+) rep=(\d+) arm=(\d+) mode=(baseline|candidate) calls=5000 seconds=([0-9.]+)',text)
 order=[(str(n),str(rep),str(arm),'candidate' if arm in (1,2) else 'baseline') for n in (64,77,256,1024,4096) for rep in range(6) for arm in range(4)]
 if [r[:4] for r in rows]!=order or any(not math.isfinite(float(r[4])) or float(r[4])<=0 for r in rows):raise RuntimeError('complete ordered batch timings required')
 results=[]
 for n in (64,77,256,1024,4096):
  samples={m:[float(r[4])/5000 for r in rows if int(r[0])==n and r[3]==m] for m in ('baseline','candidate')};means={m:sum(x)/len(x) for m,x in samples.items()};pairs=[]
  for left,right in ((0,1),(3,2)):
   b=sum(float(r[4]) for r in rows if int(r[0])==n and int(r[2])==left)/30000;c=sum(float(r[4]) for r in rows if int(r[0])==n and int(r[2])==right)/30000;pairs.append((1-c/b)*100)
  results.append(dict(columns=n,baseline_seconds=means['baseline'],candidate_seconds=means['candidate'],latency_reduction_percent=(1-means['candidate']/means['baseline'])*100,paired_reduction_percent=pairs,samples_seconds=samples))
 archive=ROOT/'build/accepted/softmax-shapes-20261009T201945Z';index=json.loads((archive/'archive-sha256.json').read_text());manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=index['build/sd-baseline-ve/manifest.json'] or sha(ROOT/'build/sd-softmax-sum-probe/baseline.o')!=index['actual-softmax-sum-object.o']:raise RuntimeError('actual accepted baseline changed')
 for n,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('model input changed')
 names=('tests/benchmark_sd_softmax_sum.cpp','scripts/benchmark_sd_softmax_sum.sh','src/ve_sd_turbo_softmax_sum.c','build/sd-softmax-sum-probe/baseline.o','build/sd-softmax-sum-probe/candidate.o','build/sd-softmax-sum-probe/probe','build/sd-baseline-ve/manifest.json','docs/results/20261009T201945Z-sd-turbo-softmax-shapes.json');files=[ROOT/n for n in names]+[log]+paths
 v=dict(status='strict_softmax_sum_batch_verified',results=results,independent_cpu_checks=108,timed_batches=120,calls_per_batch=5000,model_unchanged=True,temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(x.relative_to(ROOT)):sha(x) for x in files},publisher_sha256=sha(Path(__file__)),scope='single worker; actual O1 model object versus same-source O2 scalar strict sequential FP64 sum; synthetic nonnegative exp-domain fixtures; batch timings, not full softmax/model speedup')
 safe(v);out.write_text(json.dumps(v,indent=2)+'\n');print('Strict sum CPU audit passed:',out.relative_to(ROOT));print([(r['columns'],r['latency_reduction_percent']) for r in results])
if __name__=='__main__':main()
