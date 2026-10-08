"""Native VE greedy MTP acceptance; launch through temperature_guard.py.

Private artifacts retain sequences. Public summaries omit text and token IDs.
Existing independent CPU greedy traces are reused; this is not a new full-logit
numerical oracle. Both executors and both models are pinned before testing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from check_qwen_model import PROMPTS,chat

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--boundaries',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--accepted',type=Path)
    parser.add_argument('--depth-screen',action='store_true')
    parser.add_argument('--benchmark-case',type=int,choices=(0,1,2),default=1)
    parser.add_argument('--draft-tokens',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    print('Checking pinned executor and model checksums',flush=True)
    binary=ROOT/'build/qwen36-mtp/qwen-mtp-ve'
    target=ROOT/'build/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    head=ROOT/'build/models/qwen36-mtp/mtp-shared-q4km.gguf'
    manifest=json.loads((binary.parent/'manifest.json').read_text())
    compat=json.loads((head.parent/'compatibility.json').read_text())
    gate=json.loads((ROOT/'build/results/20261002T151124Z-qwen36/numerical-progress.json').read_text())
    if gate['status']!='correctness_complete' or not gate['cache_reset_completed']:
        raise RuntimeError('saved independent CPU reference is not accepted')
    if args.benchmark or args.depth_screen:
        if args.smoke or args.boundaries or args.accepted is None:raise RuntimeError('benchmark requires the completed acceptance suite')
        if args.benchmark and args.depth_screen:raise RuntimeError('choose one measurement mode')
        accepted=json.loads((args.accepted/'summary.json').read_text())
        if not accepted['completed'] or len(accepted['cases'])!=3 or 'perturbed_rollback' not in accepted or 'natural_end' not in accepted:
            raise RuntimeError('complete native acceptance required before benchmark')
        for key,expected in [('binary_sha256',sha(binary)),('target_sha256',compat['target_sha256']),('mtp_sha256',compat['mtp_sha256'])]:
            if accepted[key]!=expected:raise RuntimeError('benchmark artifact differs from accepted artifact')
    for path,expected in [(binary,manifest['sha256']['build/qwen36-mtp/qwen-mtp-ve']),
                          (target,compat['target_sha256']),(head,compat['mtp_sha256'])]:
        if sha(path)!=expected:raise RuntimeError('test artifact checksum mismatch')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib')
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen36-mtp',time.gmtime())
    folder.mkdir(exist_ok=False)
    report={'completed':False,'kind':'depth_screen' if args.depth_screen else 'benchmark' if args.benchmark else 'validation','test_script_sha256':sha(Path(__file__)),
            'binary_sha256':sha(binary),'target_sha256':compat['target_sha256'],
            'mtp_sha256':compat['mtp_sha256'],'threads':8,'context':512,'cache_mib':16384,'draft_tokens':args.draft_tokens,
            'benchmark_case':args.benchmark_case,
            'correctness_scope':'greedy sequence against ordinary VE and saved independent CPU traces; not full logits',
            'compatibility':compat,'cases':[]}
    def save(): (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    (folder/'check_qwen36_mtp.py').write_bytes(Path(__file__).read_bytes())
    def run(name,prompt,mode,count=12,force=True,repeats=2):
        print('Native run',name,'mode',mode,'tokens',count,'repeats',repeats,flush=True)
        command=['ve_exec','-N','1',str(binary),'--model',str(target),'--prompt',prompt,
                 '--tokens',str(count),'--threads','8','--context','512','--repeats',str(repeats)]
        if force:command+=['--force-count']
        if mode!='baseline':command+=['--mtp-model',str(head),'--draft-tokens',str(args.draft_tokens)]
        if mode=='perturbed':command+=['--reject-first']
        with (folder/(name+'.jsonl')).open('w') as out,(folder/(name+'.stderr')).open('w') as err:
            proc=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=3600)
        if proc.returncode:raise RuntimeError('native test failed: '+name)
        records=[json.loads(line) for line in (folder/(name+'.jsonl')).read_text().splitlines()]
        if len(records)!=repeats or any(r['greedy']!=records[0]['greedy'] for r in records):
            raise RuntimeError('same-process reset mismatch: '+name)
        return records
    def metrics(rows):
        warm=rows[1:] or rows
        return {'generated_tokens':rows[0]['generated_tokens'],'prompt_tokens':rows[0]['prompt_tokens'],
                'warm_tokens_per_second':statistics.median(r['tokens_per_second'] for r in warm),
                'warm_prefill_seconds':statistics.median(r['prefill_seconds'] for r in warm),
                'cold_prefill_seconds':rows[0]['prefill_seconds'],
                'samples':[{k:v for k,v in r.items() if k not in ('greedy','text')} for r in rows]}
    if args.depth_screen:
        index=args.benchmark_case
        rows=run('depth-screen',chat(PROMPTS[index]),'mtp',12,True,3)
        gold=json.loads((ROOT/('build/results/20261002T151124Z-qwen36/case%d-cpu.jsonl'%index)).read_text().splitlines()[0])['greedy_tokens']
        if any(row['greedy']!=gold for row in rows):raise RuntimeError('one-draft sequence differs from CPU reference')
        base=accepted['cases'][index]['baseline']
        report.update(case=index,candidate=metrics(rows),baseline_from_acceptance=base,
                      screening_scope='new candidate warm samples against prior acceptance baseline; not ABBA')
        report['speedup']=report['candidate']['warm_tokens_per_second']/base['warm_tokens_per_second']
        report['completed']=True;save();print('MTP depth screen PASS; artifacts',folder.relative_to(ROOT),flush=True);return
    if args.benchmark:
        reference=None;groups={'baseline':[],'mtp':[]};rates={'baseline':[],'mtp':[]}
        report['abba_order']=['baseline','mtp','mtp','baseline'];report['abba_runs']=[]
        for index,mode in enumerate(report['abba_order']):
            rows=run('abba%d-%s'%(index,mode),chat(PROMPTS[args.benchmark_case]),mode,16,True,3)
            if reference is None:reference=rows[0]['greedy']
            if any(row['greedy']!=reference for row in rows):raise RuntimeError('ABBA output sequence mismatch')
            groups[mode].extend(rows[1:]);rates[mode].extend(row['tokens_per_second'] for row in rows[1:])
            report['abba_runs'].append({'index':index,'mode':mode,**metrics(rows)});save()
            print('ABBA run',index,mode,'PASS',flush=True)
        report['warm_tokens_per_second']={key:statistics.median(values) for key,values in rates.items()}
        report['warm_prefill_seconds']={key:statistics.median(row['prefill_seconds'] for row in values) for key,values in groups.items()}
        report['speedup']=report['warm_tokens_per_second']['mtp']/report['warm_tokens_per_second']['baseline']
        report['warm_sample_counts']={key:len(values) for key,values in rates.items()}
        report['completed']=True;save();print('MTP ABBA PASS; artifacts',folder.relative_to(ROOT),flush=True);return
    for index,prompt in enumerate(PROMPTS[:1] if args.smoke else PROMPTS):
        n=8 if args.smoke else 12
        ordinary=run('case%d-baseline'%index,chat(prompt),'baseline',n)
        gold=json.loads((ROOT/('build/results/20261002T151124Z-qwen36/case%d-cpu.jsonl'%index)).read_text().splitlines()[0])['greedy_tokens'][:n]
        if ordinary[0]['greedy']!=gold:raise RuntimeError('ordinary executor differs from saved CPU greedy reference')
        mtp=run('case%d-mtp'%index,chat(prompt),'mtp',n)
        if mtp[0]['greedy']!=ordinary[0]['greedy']:raise RuntimeError('native MTP greedy mismatch')
        item={'case':index,'greedy_agreement':n,'reset_repeats':2,
              'saved_cpu_trace_sha256':sha(ROOT/('build/results/20261002T151124Z-qwen36/case%d-cpu.jsonl'%index)),
              'baseline':metrics(ordinary),'mtp':metrics(mtp)}
        item['warm_speedup']=item['mtp']['warm_tokens_per_second']/item['baseline']['warm_tokens_per_second']
        report['cases'].append(item);save();print('MTP case',index,'greedy/reset PASS, speedup',round(item['warm_speedup'],3),flush=True)
    if not args.smoke:
        perturbed=run('perturbed',chat(PROMPTS[1]),'perturbed')
        baseline=json.loads((folder/'case1-baseline.jsonl').read_text().splitlines()[0])
        if any(r['greedy']!=baseline['greedy'] or r['rollbacks']==0 for r in perturbed):
            raise RuntimeError('perturbed draft rollback sequence failed')
        report['perturbed_rollback']=metrics(perturbed);save()
        natural_base=run('natural-baseline',chat(PROMPTS[0]),'baseline',12,False)
        natural_mtp=run('natural-mtp',chat(PROMPTS[0]),'mtp',12,False)
        if natural_base[0]['greedy']!=natural_mtp[0]['greedy']:raise RuntimeError('natural EOS mismatch')
        report['natural_end']={'baseline':metrics(natural_base),'mtp':metrics(natural_mtp)};save()
    if args.boundaries:
        prompt=' x'*129
        a=run('boundary-baseline',prompt,'baseline',8)
        b=run('boundary-mtp',prompt,'mtp',8)
        if a[0]['greedy']!=b[0]['greedy']:raise RuntimeError('prefill boundary mismatch')
        report['prefill_boundary']={'baseline':metrics(a),'mtp':metrics(b)};save()
    report['completed']=True;save()
    print('MTP suite PASS; artifacts',folder.relative_to(ROOT),flush=True)

if __name__=='__main__':main()
