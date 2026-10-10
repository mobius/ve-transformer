"""Audit PNG ABBA with complete CPU-decoded RGB and deterministic file bytes."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from PIL import Image
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');text=log.read_text()
 artifacts=re.findall(r'PNG artifacts: (build/results/[^\s]+)',text)
 if text.count('PNG_COMPLETE cases=15 calls=240')!=1 or len(artifacts)!=1 or 'Temperature guard stopped command' in text:raise RuntimeError('completed native encoder suite required')
 folder=ROOT/artifacts[0];manifest_path=ROOT/'build/sd-png-probe/fixtures/manifest.json';manifest=json.loads(manifest_path.read_text());cases=manifest['cases']
 if len(cases)!=15 or [c['index'] for c in cases]!=list(range(15)):raise RuntimeError('complete fixture order required')
 if sha(ROOT/manifest['source_proof'])!=manifest['source_proof_sha256'] or sha(ROOT/'scripts/prepare_sd_png_fixtures.py')!=manifest['generator_sha256']:raise RuntimeError('fixture provenance changed')
 configs=re.findall(r'PNG_CONFIG index=(\d+) width=(\d+) height=(\d+) channels=(\d+) bytes=(\d+)',text)
 if [tuple(map(int,r)) for r in configs]!=[(c['index'],c['width'],c['height'],3,c['rgb_bytes']) for c in cases]:raise RuntimeError('actual native dimensions differ')
 records=re.findall(r'PNG_TIME index=(\d+) arm=(\d+) rep=(\d+) codec=(reference|candidate) seconds=([0-9.]+) file=([^\s]+) input_unchanged=(\d+)',text)
 order=[(i,arm,rep) for i in range(15) for arm in range(4) for rep in range(4)]
 if len(records)!=240:raise RuntimeError('240 ordered encoding records required')
 inputs={};image_files={};first_files={};times=[]
 for c in cases:
  path=ROOT/c['rgb_path'];rgb=path.read_bytes()
  if sha(path)!=c['rgb_sha256'] or len(rgb)!=c['width']*c['height']*3:raise RuntimeError('RGB fixture changed')
  inputs[c['index']]=rgb
  if 'source_png' in c and sha(ROOT/c['source_png'])!=c['source_png_sha256']:raise RuntimeError('real source image changed')
 for record,key in zip(records,order):
  i,arm,rep=key;index,actual_arm,actual_rep,codec,seconds,name,unchanged=record
  if tuple(map(int,record[:3]))!=key or codec!=('candidate' if arm in (1,2) else 'reference') or int(unchanged)!=1 or not math.isfinite(float(seconds)) or float(seconds)<=0 or name!='case%02d-arm%d-rep%d.png'%key:raise RuntimeError('encoder ordering, mode or timing differs')
  png=folder/name;c=cases[i]
  with Image.open(png) as image:
   image.load()
   if image.mode!='RGB' or image.size!=(c['width'],c['height']) or image.tobytes()!=inputs[i]:raise RuntimeError('complete independently decoded pixels differ')
  digest=sha(png)
  if i not in first_files:first_files[i]=digest
  if digest!=first_files[i]:raise RuntimeError('O1/O2 PNG file bytes differ')
  image_files[str(png.relative_to(ROOT))]=digest;times.append(dict(index=i,arm=arm,rep=rep,codec=codec,seconds=float(seconds)))
 comparisons=[]
 for c in cases:
  i=c['index'];base=[r['seconds'] for r in times if r['index']==i and r['arm'] in (0,3) and r['rep']>0];candidate=[r['seconds'] for r in times if r['index']==i and r['arm'] in (1,2) and r['rep']>0];b=sum(base)/6;t=sum(candidate)/6
  comparisons.append(dict(index=i,kind=c['kind'],width=c['width'],height=c['height'],reference_seconds=base,candidate_seconds=candidate,reference_mean_seconds=b,candidate_mean_seconds=t,latency_reduction_percent=(1-t/b)*100))
 temp=ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)',text)[-1];fans=ROOT/re.findall(r'Fan observation:.*log=([^\s]+)',text)[-1];rows=list(csv.DictReader(temp.open()))
 if not rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in rows):raise RuntimeError('unsafe thermal evidence')
 files=[log,temp,fans,manifest_path,ROOT/'src/ve_sd_turbo_png.cpp',ROOT/'tests/sd_png_reference.cpp',ROOT/'tests/benchmark_sd_png.cpp',ROOT/'scripts/benchmark_sd_png.sh',ROOT/'scripts/prepare_sd_png_fixtures.py',ROOT/'build/vendor/sd-turbo-baseline/thirdparty/stb_image_write.h',ROOT/'build/sd-png-probe/reference.o',ROOT/'build/sd-png-probe/candidate.o',ROOT/'build/sd-png-probe/probe']+[ROOT/c['rgb_path'] for c in cases]
 real=[r for r in comparisons if r['kind'].startswith('real')];b=sum(r['reference_mean_seconds'] for r in real)/len(real);t=sum(r['candidate_mean_seconds'] for r in real)/len(real)
 output=dict(status='png_encoder_microbenchmark_verified',model_speedup_measured=False,comparisons=comparisons,real_six_mean=dict(reference_seconds=b,candidate_seconds=t,latency_reduction_percent=(1-t/b)*100),native_calls=240,full_cpu_decode_checks=240,all_png_bytes_identical=True,inputs_unchanged=True,temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}|image_files,publisher_sha256=sha(Path(__file__)),compiler_flags=dict(reference='-O1 -std=c++11 -DNDEBUG -fno-fast-math -fno-associative-math',candidate='-O2 -std=c++11 -DNDEBUG -fno-fast-math -fno-associative-math -fdiag-vector=2'),scope='single VE, same STB header and default PNG settings, 9 boundary/content and 6 accepted real RGB fixtures; ABBA one warmup and three timed writes per arm, whole encoder including file I/O; independent complete CPU decoded RGB and PNG byte comparison, no model or whole-request speedup claim')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('PNG encoder suite audited:',target.relative_to(ROOT));print(output['real_six_mean'])
if __name__=='__main__':main()
