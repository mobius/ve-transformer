"""Guarded resident MTP regression; private sequences remain in build/."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from qwen_session import QwenSession
from check_qwen_model import PROMPTS,chat,ENV
from check_qwen36_mtp import sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':
        raise RuntimeError('temperature supervision required')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    folder=ROOT/'build/results'/(stamp+'-qwen36-session')
    folder.mkdir()
    binary=ROOT/'build/qwen36-mtp/qwen-mtp-ve'
    target=ROOT/'build/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    head=ROOT/'build/models/qwen36-mtp/mtp-shared-q4km.gguf'
    manifest=json.loads((binary.parent/'manifest.json').read_text())
    compat=json.loads((head.parent/'compatibility.json').read_text())
    for path,expected in [(binary,manifest['sha256']['build/qwen36-mtp/qwen-mtp-ve']),
                          (target,compat['target_sha256']),(head,compat['mtp_sha256'])]:
        if sha(path)!=expected:raise RuntimeError('artifact checksum mismatch')
    report=dict(completed=False,binary_sha256=sha(binary),target_sha256=compat['target_sha256'],
                mtp_sha256=compat['mtp_sha256'],test_sha256=sha(Path(__file__)),threads=8,context=512,
                modes=[],scope='saved independent CPU greedy traces; no full-logit comparison',
                adopted=False)
    def save():
        (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    (folder/'check_qwen36_session.py').write_bytes(Path(__file__).read_bytes())
    golden={}
    for index in range(3):
        path=ROOT/('build/results/20261002T151124Z-qwen36/case%d-cpu.jsonl'%index)
        golden[index]=json.loads(path.read_text().splitlines()[0])['greedy_tokens']
    for mode in ['ordinary','fixed','auto']:
        command=['ve_exec','-N','1',str(binary),'--model',str(target),'--threads','8',
                 '--context','512','--serve','--force-count']
        if mode!='ordinary':command+=['--mtp-model',str(head)]
        if mode=='auto':command+=['--auto-mtp']
        result=dict(mode=mode,requests=[],protocol_rejection=False)
        print('Resident test:',mode,flush=True)
        with (folder/(mode+'.stderr')).open('w') as err, (folder/(mode+'.jsonl')).open('w') as raw:
            session=QwenSession(command,err,ENV,timeout=600)
            result['host_startup_seconds']=session.startup_seconds
            try:
                for index in [1,2,0,1]:
                    row=session.request(chat(PROMPTS[index]),12,raw=True)
                    raw.write(json.dumps(row,ensure_ascii=False)+'\n');raw.flush()
                    if row['greedy']!=golden[index]:raise RuntimeError('resident CPU sequence mismatch')
                    result['requests'].append(dict(case=index,**{k:v for k,v in row.items() if k not in ('greedy','text')}))
                    print('Request verified:',mode,index,'auto_disabled',row['auto_disabled'],flush=True)
                malformed={'ordinary':struct.pack('<II',1048577,12),
                           'fixed':b'\x01\x00\x00',
                           'auto':struct.pack('<II',5,12)+b'ab'}[mode]
                session.process.stdin.write(malformed);session.process.stdin.flush();session.process.stdin.close()
                code=session.close()
                if code==0:raise RuntimeError('malformed frame was accepted')
                expected={'ordinary':'request bounds exceeded','fixed':'incomplete request header','auto':'incomplete request body'}[mode]
                err.flush()
                if expected not in (folder/(mode+'.stderr')).read_text():
                    raise RuntimeError('missing explicit protocol rejection')
                result['protocol_rejection']=True
            finally:
                session.close()
        report['modes'].append(result);save()
    report['completed']=True;save()
    print('Resident ordinary/fixed/automatic sequence, cross-request reset and malformed-frame checks PASS',flush=True)
    print('Evidence:',folder.relative_to(ROOT),flush=True)

if __name__=='__main__':main()
