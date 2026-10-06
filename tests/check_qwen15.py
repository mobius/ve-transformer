"""Qwen 1.5B accuracy and timing; launch under temperature_guard.py."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'build/models/qwen25-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf'
SHA = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
ENV = os.environ.copy()
ENV.update(VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib',OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
BINARIES = {'cpu_quant': ROOT/'build/qwen-cpu/qwen-infer',
            'cpu_accurate': ROOT/'build/qwen-cpu/qwen-infer-cpu-accurate',
            'quant': ROOT/'build/qwen15-ve/qwen-infer-quant',
            'accurate': ROOT/'build/qwen15-ve/qwen-infer-accurate'}


def chat(prompt):
    return '<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'


def run(folder,label,backend,prompt,threads=8,count=8,repeats=1,trace=False,forced=None):
    cmd=['taskset','-c','0-23'] if backend.startswith('cpu_') else ['ve_exec','-N','1']
    cmd += [str(BINARIES[backend]),'--model',str(MODEL),'--prompt',chat(prompt),
            '--threads',str(threads),'--tokens',str(count),'--context','2048',
            '--repeats',str(repeats),'--force-count']
    if not backend.startswith('cpu_'):cmd+=['--no-mmap']
    if backend=='accurate':cmd+=['--dense-cache-mib','16384']
    if trace:cmd+=['--trace',str(folder/label)]
    if forced:cmd+=['--forced-tokens',str(forced)]
    print('Running',label,flush=True)
    with (folder/(label+'.jsonl')).open('w') as out,(folder/(label+'.stderr.log')).open('w') as err:
        process=subprocess.Popen(cmd,stdout=out,stderr=err,env=ENV)
        try:code=process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();raise
    if code:raise RuntimeError(label+' failed; see ignored local stderr log')
    rows=[json.loads(line) for line in (folder/(label+'.jsonl')).read_text().splitlines()]
    assert len(rows)==repeats
    for r in rows:
        assert 1_000_000_000<r['model_parameters']<2_000_000_000 and r['generated_tokens']==count
        assert len(r['tokens'])==count and r['decode_tokens_per_second']>0
        if backend=='accurate':assert r['dense_cache_reserved_bytes']==0 and r['dense_cache_retained_bytes']<=r['dense_cache_budget_bytes']
    return rows


def main():
    digest=hashlib.sha256()
    with MODEL.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):digest.update(b)
    assert digest.hexdigest()==SHA
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen15',time.gmtime());folder.mkdir()
    report={'model':'Qwen2.5-1.5B-Instruct-Q4_K_M','model_sha256':SHA,'context':2048,'accuracy':{},'threads':{},'completed':False}
    report['binary_sha256']={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in BINARIES.items()}
    def save(): (folder/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    prompts=['Compute 17 * 23. Give only the numeric answer.','Write a Python function that adds two integers.','请用一句话说明二分查找的原理。']
    accepted=[]
    for backend in ('quant','accurate'):
        cases=[]
        for i,prompt in enumerate(prompts):
            ref_label=backend+'-cpu-'+str(i);label=backend+'-ve-'+str(i)
            ref=run(folder,ref_label,'cpu_'+backend,prompt,threads=1,trace=True)[0]
            forced=folder/(ref_label+'.tokens');forced.write_text(' '.join(map(str,ref['tokens'])))
            got=run(folder,label,backend,prompt,trace=True,forced=forced)[0]
            nv=ref['vocab'];assert got['vocab']==nv
            a=np.fromfile(folder/(ref_label+'-0.f32'),dtype='<f4').reshape(8,nv)
            b=np.fromfile(folder/(label+'-0.f32'),dtype='<f4').reshape(8,nv)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            diff=b.astype(np.float64)-a
            row=dict(max_abs_error=float(np.abs(diff).max()),rmse=float(np.sqrt(np.mean(diff**2))),argmax_agreement=int(np.sum(a.argmax(1)==b.argmax(1))),steps=8)
            row['passed']=row['max_abs_error']<=.1 and row['rmse']<=.01 and row['argmax_agreement']==8
            cases.append(row);report['accuracy'][backend]=cases;save();print(backend,i,row,flush=True)
        if all(c['passed'] for c in cases):accepted.append(backend)
    if not accepted:raise RuntimeError('neither math path passed the predetermined gates')
    candidates={}
    for backend in accepted:
        rows=run(folder,backend+'-reset',backend,prompts[1],count=32,repeats=3)
        gold=run(folder,backend+'-reset-cpu','cpu_'+backend,prompts[1],threads=1,count=32)[0]
        assert all(r['tokens']==gold['tokens'] for r in rows)
        candidates[backend]=statistics.median(r['decode_tokens_per_second'] for r in rows[1:])
    selected=max(candidates,key=candidates.get);report['selected_backend']=selected;report['accepted_backend_speeds']=candidates
    for threads in (1,2,4,8):
        rows=run(folder,'bench-'+str(threads),selected,prompts[1],threads=threads,count=32,repeats=3)
        assert all(r['tokens']==rows[0]['tokens'] for r in rows)
        report['threads'][str(threads)]={'warm_decode_tokens_per_second':statistics.median(r['decode_tokens_per_second'] for r in rows[1:]),'cold_prefill_seconds':rows[0]['prefill_seconds'],'warm_prefill_seconds':statistics.median(r['prefill_seconds'] for r in rows[1:]),'prompt_tokens':rows[0]['prompt_tokens'],'tokens':rows[0]['tokens']};save()
    sequences=[r['tokens'] for r in report['threads'].values()];assert all(x==sequences[0] for x in sequences)
    best=max(report['threads'],key=lambda k:report['threads'][k]['warm_decode_tokens_per_second']);report['selected_threads']=int(best)
    rows=run(folder,'long',selected,'A sorted list allows binary search to reduce the interval by half.\n'*8+'Explain binary search.',threads=int(best),count=16,repeats=2)
    assert rows[0]['tokens']==rows[1]['tokens']
    report['long_input']={k:rows[1][k] for k in ('prompt_tokens','prefill_seconds','decode_tokens_per_second')}
    report['completed']=True;save();(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Completed; artifacts='+str(folder.relative_to(ROOT)),flush=True)


if __name__=='__main__':main()
