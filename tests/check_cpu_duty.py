"""Check descendant throttling, exit propagation and guarded cancellation."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.cpu_duty import descendants
from scripts.temperature_guard import stop_group

def state(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rpartition(')')[2].split()[0]
    except FileNotFoundError:return None

def main():
    env=dict(os.environ,VE_TRANSFORMER_TEMPERATURE_SUPERVISED='1')
    base=[sys.executable,str(ROOT/'scripts/cpu_duty.py'),'--percent','25','--']
    sleeper="import time;time.sleep(.8)"
    child="import subprocess,sys;subprocess.run([sys.executable,'-c',%r]);sys.exit(7)"%sleeper
    proc=subprocess.Popen(base+[sys.executable,'-c',child],env=env,start_new_session=True)
    observed=False;observed_descendant=False;deadline=time.monotonic()+10
    try:
        while proc.poll() is None:
            members=descendants(proc.pid,proc.pid)
            stopped=[pid for pid in members if pid!=proc.pid and state(pid)=='T']
            observed|=bool(stopped)
            observed_descendant|=len(stopped)>=2
            if time.monotonic()>deadline:raise RuntimeError('controller timeout')
            time.sleep(.01)
        assert proc.returncode==7 and observed and observed_descendant
    finally:
        if proc.poll() is None:stop_group(proc)
    child="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"
    proc=subprocess.Popen(base+[sys.executable,'-c',child],env=env,start_new_session=True)
    members={};deadline=time.monotonic()+5
    try:
        while len(members)<3:
            members=descendants(proc.pid,proc.pid)
            if time.monotonic()>deadline:raise RuntimeError('descendants not created')
            time.sleep(.01)
        stop_group(proc)
        deadline=time.monotonic()+1
        while any(state(pid) not in (None,'Z') for pid in members) and time.monotonic()<deadline:time.sleep(.01)
        assert all(state(pid) in (None,'Z') for pid in members)
    finally:
        if proc.poll() is None:stop_group(proc)
    print('CPU duty: nested descendants paused, exit code preserved, guarded cancellation PASS')

if __name__=='__main__':main()
