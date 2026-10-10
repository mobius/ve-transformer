"""Independent bounded-memory audit of actual GroupNorm graph scaling."""
import argparse,json,re,math
from pathlib import Path
import numpy as np
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def ftz(x):
 bits=x.view('uint32').copy();small=((bits&0x7fffffff)>0)&((bits&0x7fffffff)<0x00800000);bits[small]&=0x80000000;return bits.view('float32')
def seed_states(n):
 state=0x81ab31;v=np.empty(n,dtype=np.uint64)
 for i in range(n):state=(state*1664525+1013904223)&0xffffffff;v[i]=state
 return v
def jump(n):
 a,c=1,0;ba,bc=1664525,1013904223
 while n:
  if n&1:a,c=(ba*a)&0xffffffff,(ba*c+bc)&0xffffffff
  ba,bc=(ba*ba)&0xffffffff,(bc*(ba+1))&0xffffffff;n>>=1
 return np.uint64(a),np.uint64(c)
def validate_input(raw,initial,ja,jc):
 states=initial.copy();edges=np.array([0,0x80000000,1,0x80000001,0x00800000,0x80800000],dtype=np.uint32);block=len(states)
 for offset in range(0,raw.size,block):
  size=min(block,raw.size-offset);values=((states[:size]>>8)%4001).astype(np.int32)-2000;expected=np.divide(values.astype(np.float32),np.float32(100),dtype=np.float32).view('uint32');positions=np.arange(offset,offset+size)%31;selected=positions<6;expected[selected]=edges[positions[selected]]
  if not np.array_equal(raw[offset:offset+size].view('uint32'),expected):raise RuntimeError('independent LCG input bits differ')
  states=(states*ja+jc)&np.uint64(0xffffffff)
def reference_group(x,eps):
 # Same row boundaries and sequential FP64 accumulation as the actual backend.
 row_sums=np.cumsum(x.astype(np.float64),axis=1)[:,-1];mean=np.float32(np.cumsum(row_sums)[-1]/x.size);centered=ftz(np.subtract(x,mean,dtype=np.float32));squared=ftz(np.multiply(centered,centered,dtype=np.float32));row_squares=np.cumsum(squared.astype(np.float64),axis=1)[:,-1];variance=np.float32(np.cumsum(row_squares)[-1]/x.size);scale=np.divide(np.float32(1),np.sqrt(np.add(variance,eps,dtype=np.float32),dtype=np.float32),dtype=np.float32);return ftz(np.multiply(centered,scale,dtype=np.float32))
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');out=a.output.resolve();out.relative_to(ROOT/'docs/results');text=log.read_text();paths=guarded_csv(log)
 if 'GROUP_NORM_SCALE_BATCH_PASS shapes=24 fixtures=72 checks=288 timings=576' not in text:raise RuntimeError('complete VE batch required')
 match=re.search(r'^GroupNorm scale artifacts: (build/results/[^\s]+)$',text,re.M)
 if not match:raise RuntimeError('artifact folder missing')
 folder=(ROOT/match[1]).resolve();folder.relative_to(ROOT/'build/results');probe=ROOT/'build/sd-group-norm-scale-probe';binding=json.loads((probe/'baseline.json').read_text());fixtures=json.loads((probe/'fixtures.json').read_text());cases=fixtures['cases']
 if len(cases)!=24 or [c['id'] for c in cases]!=list(range(24)):raise RuntimeError('24 bound cases required')
 teams=re.findall(r'SD_GGML_THREADS stage=unknown actual=(\d+) maximum=(\d+)',text);expected_teams=[]
 for c in cases:
  for fixture in range(3):
   expected_teams += [(str(n),str(n)) for n in (2,4,8) for _ in range(2)]
   if fixture==0:expected_teams += [('8','8')]*24
 if teams!=expected_teams:raise RuntimeError('actual parallel team trace differs')
 text=re.sub(r'SD_GGML_THREADS stage=unknown actual=\d+ maximum=\d+\n','',text);checks=re.findall(r'CHECK id=(\d+) fixture=(\d+) threads=(\d+) ne=([0-9,]+) groups=(\d+) eps=([^ ]+) bitwise_source=PASS',text);expected=[]
 for c in cases:
  for fixture in range(3):
   for n in (1,2,4,8):expected.append((c['id'],fixture,n,','.join(map(str,c['ne'])),c['groups'],int(np.float32(c['epsilon']).view('uint32'))))
 if [(int(i),int(f),int(n),ne,int(g),int(np.float32(e).view('uint32'))) for i,f,n,ne,g,e in checks]!=expected:raise RuntimeError('ordered geometry/epsilon correctness checks differ')
 files=[];cpu=[];initial=seed_states(65536);ja,jc=jump(65536)
 for c in cases:
  count=math.prod(c['ne']);raw={}
  for mode in ('input','baseline','candidate'):
   file=folder/f"case{c['id']}-{mode}.f32";files.append(file)
   if file.stat().st_size!=count*4:raise RuntimeError('raw output size differs')
   raw[mode]=np.memmap(file,dtype='<f4',mode='r')
  validate_input(raw['input'],initial,ja,jc)
  for offset in range(0,count,65536):
   if not np.array_equal(raw['baseline'][offset:offset+65536].view('uint32'),raw['candidate'][offset:offset+65536].view('uint32')):raise RuntimeError('full output bit comparison differs')
  width,height,channels,batch=c['ne'];per_group=width*height*(channels//c['groups']);maximum=0.0
  for group in range(c['groups']):
   begin=group*per_group;end=begin+per_group;x=np.asarray(raw['input'][begin:end]).reshape(-1,width);reference=reference_group(x,np.float32(c['epsilon']));actual=np.asarray(raw['candidate'][begin:end]).reshape(-1,width)
   error=float(np.max(np.abs(actual-reference)));maximum=max(maximum,error)
   if not np.all(np.isfinite(actual)) or not np.allclose(actual,reference,atol=1e-6,rtol=1e-5):raise RuntimeError('independent sequential GroupNorm reference differs')
  cpu.append(dict(case=c['id'],elements=count,groups=c['groups'],max_abs_error=maximum))
 if set(folder.glob('*.f32'))!=set(files):raise RuntimeError('unexpected raw file set')
 rows=re.findall(r'TIME id=(\d+) rep=(\d+) arm=(\d+) mode=(baseline|candidate) threads=8 seconds=([0-9.]+)',text);order=[(str(c['id']),str(rep),str(arm),'candidate' if arm in (1,2) else 'baseline') for c in cases for rep in range(6) for arm in range(4)]
 if [r[:4] for r in rows]!=order or any(not math.isfinite(float(r[4])) or float(r[4])<=0 for r in rows):raise RuntimeError('ordered positive graph timings required')
 results=[]
 for c in cases:
  selected=[r for r in rows if int(r[0])==c['id']];samples={m:[float(r[4]) for r in selected if r[3]==m] for m in ('baseline','candidate')};means={m:sum(x)/len(x) for m,x in samples.items()};pairs=[]
  for left,right in ((0,1),(3,2)):
   b=sum(float(r[4]) for r in selected if int(r[2])==left)/6;after=sum(float(r[4]) for r in selected if int(r[2])==right)/6;pairs.append((1-after/b)*100)
  results.append(dict(**c,baseline_seconds=means['baseline'],candidate_seconds=means['candidate'],latency_reduction_percent=(1-means['candidate']/means['baseline'])*100,paired_reduction_percent=pairs,samples_seconds=samples))
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=binding['manifest_sha256']:raise RuntimeError('model manifest changed')
 for n,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('model input changed')
 for n,h in binding['libraries_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('accepted graph archive changed')
 proof=ROOT/'docs/results/20261009T213215Z-sd-turbo-group-norm-shapes.json'
 if sha(proof)!=fixtures['shape_proof_sha256'] or sha(probe/'sd-ggml-cpu.c')!=binding['cloned_source_sha256'] or sha(ROOT/'build/sd-baseline-overlay/sd-ggml-cpu.c')!=binding['source_sha256']:raise RuntimeError('actual source/shape binding differs')
 files += [probe/n for n in ('baseline.json','fixtures.json','shapes.tsv','sd-ggml-cpu.c','sd-ggml-cpu.c.o','libggml-cpu.a','probe')]+[ROOT/n for n in ('scripts/prepare_sd_group_norm_scale.py','scripts/benchmark_sd_group_norm_scale.sh','tests/benchmark_sd_group_norm_scale.cpp','src/ve_sd_turbo_softmax_scale.c','build/sd-baseline-ve/manifest.json')]+[proof,log]+paths
 value=dict(status='actual_group_norm_scale_verified',results=results,cpu_validation=cpu,independent_cpu_cases=24,ve_fixture_cases=72,ve_bitwise_checks=288,timed_graphs=576,actual_parallel_team_records=1008,model_unchanged=True,temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},publisher_sha256=sha(Path(__file__)),scope='actual ggml GroupNorm graph clone, only final contiguous group scaling; same mean/variance sequential accumulation; 24 real shape/epsilon combinations with synthetic inputs; CPU reference explicit signed subnormal-result flushing, atol1e-6 rtol1e-5; no model request gain claim')
 safe(value);out.write_text(json.dumps(value,indent=2)+'\n');print('GroupNorm scale CPU audit passed:',out.relative_to(ROOT));print([(r['id'],r['latency_reduction_percent']) for r in results])
if __name__=='__main__':main()
