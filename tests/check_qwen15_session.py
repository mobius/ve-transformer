"""Run only under temperature_guard.py; compare saved CPU gold and resident resets."""
from pathlib import Path
import hashlib
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from qwen_session import QwenSession, chat


def main():
    model = ROOT/'build/models/qwen25-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf'
    h = hashlib.sha256()
    with model.open('rb') as f:
        for block in iter(lambda: f.read(16*1024**2), b''): h.update(block)
    assert h.hexdigest() == '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
    assert hashlib.sha256((ROOT/'build/qwen15-ve/qwen-infer-accurate').read_bytes()).hexdigest() == '137e8723d99121114485879a2ec29850e3c71f96c578c776b2d31dc39c7d8180'
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir',type=Path,default=ROOT/'build/results/20261006T031326Z-qwen15')
    gold = parser.parse_args().reference_dir
    assert json.loads((gold/'summary.json').read_text())['model_sha256'] == h.hexdigest()
    folder = ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen15-session',time.gmtime())
    folder.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib')
    binary = ROOT/'build/qwen15-session/qwen-infer-ve'
    command = ['ve_exec','-N','1',str(binary),'--model',str(model),'--threads','4','--context','2048','--no-mmap','--dense-cache-mib','16384','--force-count']
    report = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), completed=False, accuracy=[], timing=[])
    def save(): (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    prompts = ['Compute 17 * 23. Give only the numeric answer.','Write a Python function that adds two integers.','请用一句话说明二分查找的原理。']
    with (folder/'accuracy.stderr.log').open('w') as err:
        session = QwenSession(command+['--serve','--trace',str(folder/'trace')],err,env)
        try:
            for i,prompt in enumerate(prompts):
                row = session.request(prompt,8)
                expected = list(map(int,(gold/('accurate-cpu-'+str(i)+'.tokens')).read_text().split()))
                assert row['tokens'] == expected
                a = np.fromfile(gold/('accurate-cpu-'+str(i)+'-0.f32'),dtype='<f4').reshape(8,row['vocab'])
                b = np.fromfile(folder/('trace-'+str(i)+'.f32'),dtype='<f4').reshape(a.shape)
                diff = b.astype('float64')-a
                metrics = dict(max_abs_error=float(np.abs(diff).max()),rmse=float(np.sqrt(np.mean(diff**2))),argmax_agreement=int((a.argmax(1)==b.argmax(1)).sum()))
                assert metrics['max_abs_error']<=.1 and metrics['rmse']<=.01 and metrics['argmax_agreement']==8
                report['accuracy'].append(metrics);save();print('accuracy',i,metrics,flush=True)
        finally: assert session.close()==0
    expected32 = json.loads((gold/'accurate-reset-cpu.jsonl').read_text().splitlines()[0])['tokens']
    expected_long = json.loads((gold/'long.jsonl').read_text().splitlines()[0])['tokens']
    long_prompt = 'A sorted list allows binary search to reduce the interval by half.\n'*8+'Explain binary search.'
    with (folder/'timing.stderr.log').open('w') as err:
        session = QwenSession(command+['--serve'],err,env)
        report['startup_seconds'] = session.startup_seconds;report['ready'] = session.ready
        try:
            for i,(prompt,count,expected) in enumerate([(prompts[1],32,expected32)]*3+[(long_prompt,16,expected_long),(prompts[1],32,expected32)]):
                row = session.request(prompt,count)
                assert row['tokens']==expected and row['dense_cache_reserved_bytes']==0
                assert row['dense_cache_retained_bytes']<=row['dense_cache_budget_bytes']
                if i: assert row['cache_new_entries']==0 and row['cache_fill_thread_seconds']==0
                report['timing'].append({k:v for k,v in row.items() if k not in ('text','tokens','greedy_tokens')})
                save();print('request',i,'roundtrip',row['host_roundtrip_seconds'],'decode_tps',row['decode_tokens_per_second'],flush=True)
            try:session.request('x',1025)
            except ValueError:pass
            else:raise AssertionError('client bounds not enforced')
        finally:assert session.close()==0
    report['oneshot'] = []
    for i in range(2):
        start = time.perf_counter()
        with (folder/('oneshot-'+str(i)+'.stderr.log')).open('w') as err:
            p = subprocess.run(command+['--prompt',chat(prompts[1]),'--tokens','32'],env=env,stdout=subprocess.PIPE,stderr=err,timeout=300,check=True)
        row = json.loads(p.stdout);assert row['tokens']==expected32
        report['oneshot'].append(dict(host_seconds=time.perf_counter()-start,**{k:v for k,v in row.items() if k not in ('text','tokens','greedy_tokens')}));save()
    with (folder/'profile.stderr.log').open('w') as err:
        p = subprocess.run(command+['--prompt',chat(prompts[1]),'--tokens','8','--repeats','2','--profile'],env=env,stdout=subprocess.PIPE,stderr=err,timeout=300,check=True)
    report['profile'] = [json.loads(line[8:]) for line in (folder/'profile.stderr.log').read_text().splitlines() if line.startswith('PROFILE ')]
    report['invalid_frames'] = []
    for label,frame,message in [('header',b'\x01','incomplete request header'),('body',struct.pack('<II',4,1)+b'x','incomplete request body'),('bounds',struct.pack('<II',1024*1024+1,1),'request bounds exceeded'),('zero-count',struct.pack('<II',0,0),'request bounds exceeded'),('context',struct.pack('<II',8192,1)+b'a '*4096,'prompt exceeds requested context')]:
        with (folder/(label+'.stderr.log')).open('w') as err:
            session = QwenSession(command+['--serve'],err,env)
            session.process.stdin.write(frame);session.process.stdin.flush()
            code=session.close();assert code!=0
            assert message in (folder/(label+'.stderr.log')).read_text()
            report['invalid_frames'].append(dict(case=label,exit_code=code))
    report['completed']=True;save();print('Completed',folder.relative_to(ROOT),flush=True)


if __name__=='__main__':main()
