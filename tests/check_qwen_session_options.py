"""Check client engine routing and chat framing without launching hardware."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import qwen_session

class FakeSession:
    calls=[]
    def __init__(self,command,stderr,env):
        self.ready={'event':'ready','protocol':1};self.startup_seconds=0
        self.command=command;self.requests=[];self.calls.append(self)
    def request(self,prompt,count,raw=False):
        self.requests.append((prompt,count,raw));return {'generated_tokens':count}
    def close(self):return 0

def main():
    with tempfile.TemporaryDirectory() as folder:
        base=['client','--model','local.gguf','--prompt','hello','--tokens','12',
              '--executor','test-ve','--stderr-log',str(Path(folder)/'stderr')]
        for extra in [[],['--engine','qwen36'],['--engine','qwen36','--mtp-model','head.gguf'],
                      ['--engine','qwen36','--mtp-model','head.gguf','--auto-mtp']]:
            with patch.object(sys,'argv',base+extra),patch.object(qwen_session,'QwenSession',FakeSession),contextlib.redirect_stdout(io.StringIO()):
                qwen_session.main()
            session=FakeSession.calls[-1]
            modern='qwen36' in extra
            assert ('--no-mmap' in session.command)==(not modern)
            assert ('--auto-mtp' in session.command)==('--auto-mtp' in extra)
            assert ('--mtp-model' in session.command)==('--mtp-model' in extra)
            prompt,count,raw=session.requests[0]
            assert count==12 and raw==modern
            if modern:assert prompt=='<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            else:assert prompt=='hello'
        for args in [base+['--mtp-model','head.gguf'],base+['--engine','qwen36','--auto-mtp'],
                     ['client','--engine','qwen36','--model','local.gguf','--prompt','hello']]:
            with patch.object(sys,'argv',args),patch.object(qwen_session,'QwenSession',FakeSession),contextlib.redirect_stderr(io.StringIO()):
                try:qwen_session.main()
                except SystemExit as exc:assert exc.code==2
                else:raise AssertionError('invalid engine option accepted')
    print('Client engine routing, non-thinking chat and invalid options PASS')

if __name__=='__main__':main()
