"""Controlled native allocation verifies the local read-only memory prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import time
from ve_memory_info import VeMemoryInfo
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=(1,2,3),default=1);args=p.parse_args()
 reader=VeMemoryInfo();folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-ve-memory-calibration',time.gmtime());folder.mkdir()
 executable=ROOT/'build/ve-memory-probe'
 report={'completed':False,'runtime_slot':args.slot,'library_version':reader.version,'allocation_bytes':512*1024**2,'allocation_kind':'anonymous_mmap_with_page_touch',
         'source_sha256':hashlib.sha256((ROOT/'src/ve_memory_probe.c').read_bytes()).hexdigest(),
         'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'adapter_sha256':hashlib.sha256((ROOT/'scripts/ve_memory_info.py').read_bytes()).hexdigest(),
         'prefix_scope':'three unsigned-long output fields only; oversized output; not a complete structure ABI claim',
         'memory_scope':'VEOS whole-node reported usage, includes runtime and other processes; not model peak',
         'binary_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),'samples':[]}
 def save():(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 report['before_process']=reader.sample(args.slot)
 save();env=dict(os.environ,VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib')
 with (folder/'native.stderr').open('w') as error:
  child=subprocess.Popen(['ve_exec','-N',str(args.slot),str(executable)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True)
  try:
   for stage in ('baseline','allocated','released'):
    if not select.select([child.stdout],[],[],60)[0]:raise RuntimeError('native handshake timeout')
    line=child.stdout.readline()
    match=re.fullmatch(r'VE_MEMORY_STAGE (\w+) total_pages=(-?\d+) free_pages=(-?\d+) page_bytes=(-?\d+) maxrss_kib=(-?\d+)\n',line)
    if not match or match[1]!=stage:raise RuntimeError('native allocation protocol failed')
    row={'stage':stage,'node_memory':reader.sample(args.slot),'native_sysconf':dict(zip(('total_pages','free_pages','page_bytes','maxrss_kib'),map(int,match.groups()[1:])))}
    report['samples'].append(row);save()
    if stage=='released':
     report['release_poll_samples']=[]
     start=time.monotonic()
     for _ in range(20):
      time.sleep(.5)
      report['release_poll_samples'].append({'elapsed_seconds':time.monotonic()-start,'node_memory':reader.sample(args.slot)})
      save()
    child.stdin.write('a');child.stdin.flush()
   if child.wait(timeout=30):raise RuntimeError('native allocation probe failed')
  finally:
   if child.poll() is None:child.kill();child.wait()
 report['after_exit_poll_samples']=[]
 start=time.monotonic()
 for _ in range(10):
  time.sleep(.5)
  report['after_exit_poll_samples'].append({'elapsed_seconds':time.monotonic()-start,'node_memory':reader.sample(args.slot)})
 save()
 baseline,allocated,released=[r['node_memory']['used_kib'] for r in report['samples']]
 delta=allocated-baseline;remaining=released-baseline
 native=report['samples'][0]['native_sysconf'];page_kib=native['page_bytes']//1024
 if native['total_pages']*native['page_bytes']!=report['samples'][0]['node_memory']['total_kib']*1024 or page_kib<=0:
  raise RuntimeError('native page geometry differs from VEOS memory domain')
 report.update(allocation_delta_kib=delta,released_delta_kib=remaining,page_rounding_allowance_kib=page_kib)
 save()
 if not 512*1024<=delta<=512*1024+page_kib:
  raise RuntimeError('controlled allocation delta differs from expected size')
 if any(abs(r['node_memory']['used_kib']-baseline)>page_kib for r in report['release_poll_samples'][-3:]):
  raise RuntimeError('unmapped pages did not return to process baseline within ten seconds')
 if any(abs(r['node_memory']['used_kib']-report['before_process']['used_kib'])>page_kib for r in report['after_exit_poll_samples'][-3:]):
  raise RuntimeError('process exit did not return node usage to initial baseline')
 settled=report['release_poll_samples'][-1]['node_memory']['used_kib']-baseline
 report.update(completed=True,immediate_released_delta_kib=remaining,settled_released_delta_kib=settled,
               validated_prefix_for_local_version=True)
 save();print('VE memory calibration passed:',folder.relative_to(ROOT),'allocation_delta_kib',delta,'settled_released_delta_kib',settled,flush=True)
if __name__=='__main__':main()
