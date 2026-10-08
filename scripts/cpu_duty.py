"""Limit only a command's CPU duty cycle; use inside temperature_guard.py.

No global frequency, affinity policy or fan changes. Descendants stay in the
guarded process group. Inspect process identity before signaling to avoid PID
reuse. Never read command lines or other potentially sensitive process data.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

def identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_bytes().rpartition(b')')[2].split()
        return int(fields[1]),int(fields[2]),int(fields[19]) # parent, group, start ticks
    except (OSError,ValueError,IndexError):return None

def descendants(root,group):
    table={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            value=identity(int(path.name))
            if value and value[1]==group:table[int(path.name)]=value
    found={root} if root in table else set()
    while True:
        more={pid for pid,value in table.items() if value[0] in found}
        if more<=found:break
        found.update(more)
    return {pid:table[pid] for pid in found}

def send(members,sig):
    for pid,value in members.items():
        current=identity(pid)
        if current and current[1:]==value[1:]:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--percent',type=int,default=25)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command[:1]==['--']:args.command=args.command[1:]
    if not 5<=args.percent<=100 or not args.command:parser.error('percent 5..100 and a command required')
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':
        parser.error('run from the supervised build script')
    def interrupted(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,interrupted)
    proc=subprocess.Popen(args.command) # Keep the temperature guard's process group.
    group=os.getpgid(proc.pid);suspended={}
    try:
        if args.percent==100:
            code=proc.wait();return code if code>=0 else 128-code
        period=.2;active=period*args.percent/100
        while proc.poll() is None:
            time.sleep(active)
            suspended=descendants(proc.pid,group)
            send(suspended,signal.SIGSTOP)
            time.sleep(period-active)
            send(suspended,signal.SIGCONT);suspended={}
        return proc.returncode if proc.returncode>=0 else 128-proc.returncode
    finally:
        send(suspended,signal.SIGCONT)
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=1)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':raise SystemExit(main())
