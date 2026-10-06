"""Compare real-model intermediate tensors; invoke through the thermal guard."""
import collections
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from check_qwen_model import ROOT,MODEL,ENV,PROMPTS,chat

parser=argparse.ArgumentParser()
parser.add_argument('--analyze-folder',type=Path,help='reuse completed traces without rerunning hardware')
parser.add_argument('--math',choices=['quant','float','accurate'],default='quant')
args=parser.parse_args()
folder=args.analyze_folder or ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen-nodes',time.gmtime())
folder=folder.resolve()
if not args.analyze_folder:folder.mkdir(parents=True)
binaries=[] if args.analyze_folder else [('cpu',ROOT/'build/qwen-cpu/qwen-infer'),
                     ('ve',ROOT/'build/qwen-infer-ve'),
                     ('baseline',ROOT/'build/qwen-infer-ve-baseline')]
if args.math=='float' and not args.analyze_folder:
    binaries=[('cpu',ROOT/'build/qwen-cpu/qwen-infer-cpu-float'),('ve',ROOT/'build/qwen-infer-ve-nlc')]
if args.math=='accurate' and not args.analyze_folder:
    binaries=[('cpu',ROOT/'build/qwen-cpu/qwen-infer-cpu-accurate'),('ve',ROOT/'build/qwen-ve-accurate/qwen-infer-ve')]
for label,binary in binaries:
    command=['taskset','-c','0-23'] if label=='cpu' else ['ve_exec','-N','1']
    command += [str(binary),'--model',str(MODEL),'--prompt',chat(PROMPTS[0]),
                '--tokens','1','--threads','1' if label=='cpu' else '8','--context','4096',
                '--force-count','--node-trace',str(folder/label),'--trace',str(folder/label)]
    if label!='cpu':command+=['--no-mmap']
    print('Tracing '+label,flush=True)
    with (folder/(label+'.stderr.log')).open('w') as err:
        result=subprocess.run(command,env=ENV,stdout=subprocess.PIPE,stderr=err,text=True,
                              timeout=1800,check=True)
    (folder/(label+'.jsonl')).write_text(result.stdout)
    print(result.stdout,flush=True)

def records(label):
    counts=collections.Counter();rows={}
    for line in (folder/(label+'.nodes.jsonl')).read_text().splitlines():
        row=json.loads(line)
        key=(row['phase'],row['repeat'],row['name'],row['op'],row['type'],tuple(row['ne']))
        counts[key]+=1;rows[(key,counts[key])]=row
    return rows

reference=records('cpu');reports={}
for label in ('ve','baseline'):
    if not (folder/(label+'.nodes.jsonl')).exists():continue
    actual=records(label);differences=[];unmatched=[]
    with (folder/'cpu.nodes.bin').open('rb') as a,(folder/(label+'.nodes.bin')).open('rb') as b:
        for key,row in reference.items():
            other=actual.get(key)
            if other is None:unmatched.append(row);continue
            if row['bytes']!=other['bytes']:raise RuntimeError('matched node byte size changed')
            if row['bytes']==0:continue
            a.seek(row['offset']);b.seek(other['offset'])
            dtype='<f4' if row['type']=='f32' else '<i4'
            ref=np.frombuffer(a.read(row['bytes']),dtype=dtype)
            got=np.frombuffer(b.read(other['bytes']),dtype=dtype)
            if ref.nbytes!=row['bytes'] or got.nbytes!=other['bytes']:
                raise RuntimeError('incomplete node trace payload')
            delta=got.astype(np.float64)-ref
            maximum=float(np.max(np.abs(delta)))
            if maximum:
                differences.append(dict(phase=row['phase'],name=row['name'],op=row['op'],
                    type=row['type'],ne=row['ne'],max_abs_error=maximum,
                    rmse=float(np.sqrt(np.mean(delta**2))),
                    reference_max=float(np.max(np.abs(ref.astype(np.float64)))),
                    finite=bool(np.isfinite(ref).all() and np.isfinite(got).all())))
    reports[label]=dict(differences=differences,unmatched_reference=unmatched,
                       reference_nodes=len(reference),actual_nodes=len(actual))
    significant=[row for row in differences if row['type']=='i32' or
                 row['max_abs_error']>0.001+0.001*row['reference_max']]
    print(label+' FIRST_SIGNIFICANT '+json.dumps(significant[:30]),flush=True)
(folder/'node-differences.json').write_text(json.dumps(reports,indent=2)+'\n')
print('artifacts='+str(folder.relative_to(ROOT)),flush=True)
