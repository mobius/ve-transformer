"""Wait for a measured cool starting point inside the existing temperature guard."""
import argparse,json,os,time
from pathlib import Path
from temperature_guard import discover,sample
ROOT=Path(__file__).resolve().parents[1]
def main():
 if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':raise RuntimeError('temperature supervision required')
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-max',type=float,default=68);p.add_argument('--ve-max',type=float,default=56);p.add_argument('--stable-seconds',type=float,default=10);p.add_argument('--timeout',type=float,default=300);a=p.parse_args()
 output=a.output.resolve();output.relative_to(ROOT/'build')
 if not 0<a.cpu_max<80 or not 0<a.ve_max<70 or not 0<a.stable_seconds<a.timeout<=600:raise RuntimeError('valid cool-start limits required')
 channels=discover();start=time.monotonic();stable=None
 value=dict(ready=False,cpu_max_c=a.cpu_max,ve_max_c=a.ve_max,stable_seconds=a.stable_seconds,timeout_seconds=a.timeout)
 while True:
  readings=sample(channels);cpu=max(v for k,v in readings.items() if k.startswith('cpu/'));ve=max(v for k,v in readings.items() if k.startswith('ve'))
  now=time.monotonic()
  if cpu<=a.cpu_max and ve<=a.ve_max:
   if stable is None:stable=now
   if now-stable>=a.stable_seconds:
    value.update(ready=True,elapsed_seconds=now-start,cpu_c=cpu,ve_c=ve);output.write_text(json.dumps(value,indent=2)+'\n');print('Measured cool start ready:',cpu,ve,'seconds',round(now-start,3));return
  else:stable=None
  if now-start>=a.timeout:
   value.update(elapsed_seconds=now-start,cpu_c=cpu,ve_c=ve);output.write_text(json.dumps(value,indent=2)+'\n');raise RuntimeError('cool starting point timeout; no inference launched')
  time.sleep(.5)
if __name__=='__main__':main()
