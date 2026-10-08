"""Thermally supervised accuracy and alternating same-load VE comparisons."""
from pathlib import Path
import argparse,hashlib,json,os,statistics,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from qwen_session import QwenSession


def main():
    model=ROOT/'build/models/qwen25-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf'
    digest=hashlib.sha256()
    with model.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):digest.update(block)
    assert digest.hexdigest()=='6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen15-attention',time.gmtime());folder.mkdir()
    gold=ROOT/'build/results/20261006T031326Z-qwen15'
    env=dict(os.environ,OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib')
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-dir',type=Path,default=ROOT/'build/qwen15-attention')
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'build/qwen15-session')
    parser.add_argument('--threads',type=int,choices=(1,2,4,8),default=4)
    parser.add_argument('--baseline-threads',type=int,choices=(1,2,4,8),default=4)
    parser.add_argument('--order',choices=('ab','abba'),default='abba')
    args=parser.parse_args()
    binaries={'baseline':args.baseline_dir/'qwen-infer-ve','attention':args.candidate_dir/'qwen-infer-ve'}
    baseline_manifest=json.loads((args.baseline_dir/'manifest.json').read_text())
    baseline_key=str(binaries['baseline'].resolve().relative_to(ROOT))
    assert hashlib.sha256(binaries['baseline'].read_bytes()).hexdigest()==baseline_manifest['sha256'][baseline_key]
    report={'completed':False,'model_sha256':digest.hexdigest(),'binary_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},'threads':args.threads,'baseline_threads':args.baseline_threads,'accuracy':[],'benchmarks':[],'candidate_directory':str(args.candidate_dir.resolve().relative_to(ROOT))}
    def save():(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def close_session(session):
        code=session.close()
        if code and sys.exc_info()[0] is None:
            raise RuntimeError('native executor exited unsuccessfully; inspect local stderr log')
    def command(kind):return ['ve_exec','-N','1',str(binaries[kind]),'--model',str(model),'--threads',str(args.baseline_threads if kind=='baseline' else args.threads),'--context','2048','--no-mmap','--dense-cache-mib','16384','--force-count','--serve']
    prompts=['Compute 17 * 23. Give only the numeric answer.','Write a Python function that adds two integers.','请用一句话说明二分查找的原理。']
    with (folder/'accuracy.stderr.log').open('w') as err:
        session=QwenSession(command('attention')+['--trace',str(folder/'trace')],err,env)
        try:
            for i,prompt in enumerate(prompts):
                row=session.request(prompt,8)
                expected=list(map(int,(gold/('accurate-cpu-'+str(i)+'.tokens')).read_text().split()))
                assert row['tokens']==expected
                a=np.fromfile(gold/('accurate-cpu-'+str(i)+'-0.f32'),dtype='<f4').reshape(8,row['vocab'])
                b=np.fromfile(folder/('trace-'+str(i)+'.f32'),dtype='<f4').reshape(a.shape)
                diff=b.astype('float64')-a
                metrics={'max_abs_error':float(np.abs(diff).max()),'rmse':float(np.sqrt(np.mean(diff**2))),'argmax_agreement':int((a.argmax(1)==b.argmax(1)).sum())}
                assert metrics['max_abs_error']<=.1 and metrics['rmse']<=.01 and metrics['argmax_agreement']==8
                report['accuracy'].append(metrics);save();print('accuracy',i,metrics,flush=True)
        finally:close_session(session)
    expected32=json.loads((gold/'accurate-reset-cpu.jsonl').read_text().splitlines()[0])['tokens']
    long_prompt='A sorted list allows binary search to reduce the interval by half.\n'*8+'Explain binary search.'
    expected_long=json.loads((gold/'long.jsonl').read_text().splitlines()[0])['tokens']
    # ABBA order reduces drift; each process includes a cold request followed by warm ones.
    for run,kind in enumerate(('baseline','attention') if args.order=='ab' else ('baseline','attention','attention','baseline')):
        print('benchmark',run,kind,flush=True)
        with (folder/('bench-'+str(run)+'.stderr.log')).open('w') as err:
            session=QwenSession(command(kind),err,env)
            rows=[]
            try:
                for i in range(4):
                    row=session.request(prompts[1],32);assert row['tokens']==expected32
                    assert row['dense_cache_reserved_bytes']==0 and row['dense_cache_retained_bytes']<=row['dense_cache_budget_bytes']
                    if i:assert row['cache_new_entries']==0
                    rows.append({k:v for k,v in row.items() if k not in ('tokens','greedy_tokens','text')})
                    print('request',run,i,'tps',row['decode_tokens_per_second'],flush=True)
                long_row=session.request(long_prompt,16);assert long_row['tokens']==expected_long
                back=session.request(prompts[1],32);assert back['tokens']==expected32 and back['cache_new_entries']==0
            finally:close_session(session)
        record={'backend':kind,'startup_seconds':session.startup_seconds,'requests':rows,'long_request':{k:v for k,v in long_row.items() if k not in ('tokens','greedy_tokens','text')},'after_long':{k:v for k,v in back.items() if k not in ('tokens','greedy_tokens','text')}}
        report['benchmarks'].append(record);save()
        print(kind,'warm tps',statistics.median(x['decode_tokens_per_second'] for x in rows[1:]),'prefill',statistics.median(x['prefill_seconds'] for x in rows[1:]),flush=True)
    report['medians']={}
    for kind in binaries:
        rows=[row for run in report['benchmarks'] if run['backend']==kind for row in run['requests'][1:]]
        report['medians'][kind]={k:statistics.median(x[k] for x in rows) for k in ('host_roundtrip_seconds','prefill_seconds','decode_seconds','decode_tokens_per_second','tokenize_seconds')}
    report['completed']=True;save();print('Completed',folder.relative_to(ROOT),report['medians'],flush=True)


if __name__=='__main__':main()
