"""Audit actual ggml Softmax graph outputs, shape binding and paired timings."""
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
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');out=a.output.resolve();out.relative_to(ROOT/'docs/results');text=log.read_text();paths=guarded_csv(log)
 teams=re.findall(r'SD_GGML_THREADS stage=unknown actual=([0-9]+) maximum=([0-9]+)',text);expected_teams=[(str(n),str(n)) for _ in range(45) for n in (2,4,8) for _ in range(2)]+[('8','8')]*260
 if teams!=expected_teams:raise RuntimeError('actual parallel team trace differs')
 text=re.sub(r'SD_GGML_THREADS stage=unknown actual=[0-9]+ maximum=[0-9]+\n','',text)
 if 'SOFTMAX_GRAPH_BATCH_PASS validation_cases=45 validation_checks=180 actual_shapes=10 timings=240' not in text:raise RuntimeError('complete actual graph batch required')
 match=re.search(r'^Softmax graph artifacts: (build/results/[^\s]+)$',text,re.M)
 if not match:raise RuntimeError('artifact folder absent')
 folder=(ROOT/match[1]).resolve();folder.relative_to(ROOT/'build/results');checks=re.findall(r'CHECK id=(\d+) ne=([0-9,]+) scale=([^ ]+) mask=(\d+) threads=(\d+) mode=(actual|validation) bitwise=PASS',text);expected=[];case=0;files=[];cpu=[]
 for n in (64,77,256,1024,4096):
  state=0x81ab31;source=[]
  for i in range(n*17):
   state=(state*1664525+1013904223)&0xffffffff;v=np.float32(np.float32(((state>>8)%4001)-2000)/np.float32(100))
   if i%31==0:v=np.float32(-100)
   if i%37==0:v=np.float32(-1e-38)
   source.append(v)
  source=np.array(source,dtype=np.float32).reshape(17,n)
  for scale in (np.float32(1),np.float32(.1),np.float32(.000244140625)):
   for mask in (0,1,2):
    for nth in (1,2,4,8):expected.append((case,f'{n},17,1,1',int(scale.view('uint32')),mask,nth,'validation'))
    raw={}
    for mode in ('input','baseline','candidate'):
     f=folder/f'case{case}-{mode}.f32';files.append(f);raw[mode]=np.fromfile(f,dtype='<f4')
     if raw[mode].size!=n*17:raise RuntimeError('raw shape differs')
    if not np.array_equal(raw['input'].view('uint32'),source.ravel().view('uint32')):raise RuntimeError('independent input differs')
    if not np.array_equal(raw['baseline'].view('uint32'),raw['candidate'].view('uint32')):raise RuntimeError('actual graph baseline/candidate bits differ')
    with np.errstate(under='ignore',invalid='ignore'):
     logits=ftz(np.multiply(source,scale,dtype=np.float32))
     if mask:
      row=np.array([np.float32(-np.inf) if j%13==12 else np.float32((j%7)*-.125) for j in range(n)],dtype=np.float32);logits=np.add(logits,row,dtype=np.float32)
     shifted=np.subtract(logits,logits.max(axis=1,keepdims=True),dtype=np.float32);exp=ftz(np.exp(shifted,dtype=np.float32));sums=np.cumsum(exp.astype(np.float64),axis=1)[:,-1];norm=(1/sums).astype(np.float32);reference=ftz(np.multiply(exp,norm[:,None],dtype=np.float32))
    actual=raw['candidate'].reshape(17,n);error=float(np.max(np.abs(actual-reference)));row_error=float(np.max(np.abs(actual.astype(np.float64).sum(axis=1)-1)))
    if not np.all(np.isfinite(actual)) or np.any(actual<0) or not np.allclose(actual,reference,atol=1e-6,rtol=1e-5) or row_error>2e-6:raise RuntimeError('independent probability reference differs')
    if mask and np.any(actual[:,np.arange(n)%13==12]!=0):raise RuntimeError('masked outputs nonzero')
    cpu.append(dict(case=case,columns=n,scale=float(scale),mask=mask,max_abs_error=error,row_sum_error=row_error));case+=1
 probe=ROOT/'build/sd-softmax-graph-probe';binding=json.loads((probe/'baseline.json').read_text())
 for i,ne in enumerate(binding['shapes']):expected.append((100+i,','.join(map(str,ne)),int(np.float32(1).view('uint32')),0,8,'actual'))
 if [(int(i),ne,int(np.float32(s).view('uint32')),int(m),int(t),mode) for i,ne,s,m,t,mode in checks]!=expected:raise RuntimeError('complete ordered graph checks required')
 if set(folder.glob('*.f32'))!=set(files):raise RuntimeError('unexpected raw file set')
 rows=re.findall(r'TIME id=(\d+) rep=(\d+) arm=(\d+) mode=(baseline|candidate) threads=8 seconds=([0-9.]+)',text);order=[(str(100+i),str(rep),str(arm),'candidate' if arm in (1,2) else 'baseline') for i in range(10) for rep in range(6) for arm in range(4)]
 if [r[:4] for r in rows]!=order or any(not math.isfinite(float(r[4])) or float(r[4])<=0 for r in rows):raise RuntimeError('complete ordered timings required')
 results=[]
 for i,ne in enumerate(binding['shapes']):
  selected=[r for r in rows if int(r[0])==100+i];samples={m:[float(r[4]) for r in selected if r[3]==m] for m in ('baseline','candidate')};means={m:sum(x)/len(x) for m,x in samples.items()};pairs=[]
  for left,right in ((0,1),(3,2)):
   b=sum(float(r[4]) for r in selected if int(r[2])==left)/6;c=sum(float(r[4]) for r in selected if int(r[2])==right)/6;pairs.append((1-c/b)*100)
  results.append(dict(shape=ne,baseline_seconds=means['baseline'],candidate_seconds=means['candidate'],latency_reduction_percent=(1-means['candidate']/means['baseline'])*100,paired_reduction_percent=pairs,samples_seconds=samples))
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=binding['manifest_sha256']:raise RuntimeError('model manifest changed')
 for n,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('model input changed')
 for n,h in binding['libraries_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('baseline actual archive changed')
 if sha(probe/'sd-ggml-cpu.c')!=binding['cloned_source_sha256'] or sha(ROOT/'build/sd-baseline-overlay/sd-ggml-cpu.c')!=binding['source_sha256']:raise RuntimeError('source binding changed')
 files += [probe/n for n in ('baseline.json','shapes.tsv','sd-ggml-cpu.c','sd-ggml-cpu.c.o','scale.o','libggml-cpu.a','probe')]+[ROOT/n for n in ('scripts/prepare_sd_softmax_graph.py','scripts/benchmark_sd_softmax_graph.sh','tests/benchmark_sd_softmax_graph.cpp','src/ve_sd_turbo_softmax_scale.c','build/sd-baseline-ve/manifest.json','docs/results/20261009T201945Z-sd-turbo-softmax-shapes.json')]+[log]+paths
 v=dict(status='actual_softmax_graph_verified',results=results,cpu_validation=cpu,independent_cpu_cases=45,ve_validation_checks=180,actual_shape_checks=10,actual_parallel_team_records=530,timed_graphs=240,model_unchanged=True,temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(x.relative_to(ROOT)):sha(x) for x in files},publisher_sha256=sha(Path(__file__)),scope='actual ggml Softmax graph cloned backend with same scalar/vector objects and runtime scale selector; real ten geometries timing at eight threads; independent probability reference uses float32 result flushing, atol1e-6 rtol1e-5; no model request gain claim')
 safe(v);out.write_text(json.dumps(v,indent=2)+'\n');print('Actual Softmax graph CPU audit passed:',out.relative_to(ROOT));print([(r['shape'],r['latency_reduction_percent']) for r in results])
if __name__=='__main__':main()
