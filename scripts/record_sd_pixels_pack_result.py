"""Audit full CPU float/RGB outputs and raw VE pixel ABBA timings."""
import argparse,csv,json,math,re
from pathlib import Path
import numpy as np
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def audit_direct_pack(folder,text):
 path=ROOT/'build/sd-pixels-pack-probe/fixtures/direct-manifest.json';manifest=json.loads(path.read_text());cases=manifest['cases'];files=[path]
 if len(cases)!=12 or [c['index'] for c in cases]!=list(range(12)) or manifest['generator_sha256']!=sha(ROOT/'scripts/prepare_sd_pixels_pack_fixtures.py') or manifest['half_threshold_groups']!=255 or manifest['neighbors_per_threshold']!=5 or text.count('DIRECT_PACK_COMPLETE cases=12 timed_calls=192 PASS')!=1:raise RuntimeError('complete direct fixture suite required')
 raw=re.findall(r'DIRECT_PACK_CHECK index=(\d+) spatial=(\d+) full_rgb_bytes=(\d+) input_unchanged=(\d+) canaries=(\d+)',text)
 if [tuple(map(int,r)) for r in raw]!=[(c['index'],c['spatial'],1,1,1) for c in cases]:raise RuntimeError('direct native checks differ')
 for c in cases:
  source=ROOT/c['input_path'];data=np.fromfile(source,dtype=np.float32)
  if sha(source)!=c['input_sha256'] or data.size!=c['spatial']*3 or not np.isfinite(data).all() or not ((data>=0)&(data<=1)).all():raise RuntimeError('direct input provenance/domain differs')
  if c['index']==0:
   if c['kind']!='half_threshold_neighbors' or data.size!=1275:raise RuntimeError('direct half-threshold shape differs')
   values=data.reshape(5,255);threshold=np.arange(255,dtype=np.float32)+np.float32(.5);scaled=values*np.float32(255)
   if not ((scaled[0]<threshold).all() and (scaled[2]==threshold).all() and (scaled[4]>threshold).all()):raise RuntimeError('all 255 half-threshold classes required')
   for before,after,direction in ((2,1,-np.inf),(1,0,-np.inf),(2,3,np.inf),(3,4,np.inf)):
    if not np.array_equal(np.nextafter(values[before],np.float32(direction)),values[after]):raise RuntimeError('direct neighboring float values differ')
  scaled=(data.reshape(3,c['spatial'])*np.float32(255)).astype(np.float64);expected=np.floor(scaled+.5).astype(np.uint8).T.copy().tobytes();rgb=folder/('direct%02d.rgb'%c['index'])
  if rgb.read_bytes()!=expected:raise RuntimeError('direct CPU full RGB bytes differ: '+c['kind'])
  files.extend([source,rgb])
 raw=re.findall(r'DIRECT_PACK_TIME index=(\d+) arm=(\d+) rep=(\d+) kernel=(reference|candidate) seconds=([0-9.]+) full_rgb_bytes=(\d+) input_unchanged=(\d+) canaries=(\d+)',text);order=[(c['index'],arm,rep) for c in cases for arm in range(4) for rep in range(4)]
 if len(raw)!=192:raise RuntimeError('192 ordered direct timings required')
 records=[]
 for r,key in zip(raw,order):
  if tuple(map(int,r[:3]))!=key or r[3]!=('candidate' if key[1] in (1,2) else 'reference') or r[5:]!=('1','1','1') or not math.isfinite(float(r[4])) or float(r[4])<=0:raise RuntimeError('direct timing/mode/invariants differ')
  records.append(dict(index=key[0],arm=key[1],rep=key[2],seconds=float(r[4])))
 comparisons=[]
 for c in cases:
  samples={name:[r['seconds'] for r in records if r['index']==c['index'] and r['arm'] in arms and r['rep']>0] for name,arms in (('reference',(0,3)),('candidate',(1,2)))};b=sum(samples['reference'])/6;t=sum(samples['candidate'])/6
  comparisons.append(dict(index=c['index'],kind=c['kind'],spatial=c['spatial'],reference_samples_seconds=samples['reference'],candidate_samples_seconds=samples['candidate'],reference_seconds=b,candidate_seconds=t,latency_reduction_percent=(1-t/b)*100))
 return comparisons,files

def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');text=log.read_text()
 artifacts=re.findall(r'Pixel artifacts: (build/results/[^\s]+)',text)
 if len(artifacts)!=1 or text.count('PIXEL_COMPLETE cases=43 finite=20 timed_pairs=320 invalid=23')!=1 or text.count('PIXEL_INVALID_ARGUMENTS checks=7 PASS')!=1 or 'Temperature guard stopped command' in text:raise RuntimeError('complete native pixel suite required')
 folder=ROOT/artifacts[0];manifest_path=ROOT/'build/sd-pixels-pack-probe/fixtures/manifest.json';manifest=json.loads(manifest_path.read_text());cases=manifest['cases']
 if len(cases)!=43 or [c['index'] for c in cases]!=list(range(43)) or sha(ROOT/manifest['reference_summary'])!=manifest['reference_summary_sha256'] or sha(ROOT/'scripts/prepare_sd_pixels_pack_fixtures.py')!=manifest['generator_sha256']:raise RuntimeError('fixture provenance differs')
 configs=re.findall(r'PIXEL_CHECK index=(\d+) spatial=(\d+) invalid=(-?\d+) clamp_ok=(\d+) full_float_bytes=(\d+) full_rgb_bytes=(\d+) canaries=(\d+)',text)
 wanted=[(c['index'],c['spatial'],c.get('invalid_index',-1),int('invalid_index' not in c),1,int('invalid_index' not in c),1) for c in cases]
 if [tuple(map(int,r)) for r in configs]!=wanted:raise RuntimeError('actual native checks differ')
 files=[log,manifest_path];output_hash={};cpu_float=cpu_rgb=0
 for c in cases:
  source=ROOT/c['input_path']
  if sha(source)!=c['input_sha256']:raise RuntimeError('input changed')
  data=np.fromfile(source,dtype=np.float32)
  if data.size!=c['spatial']*3:raise RuntimeError('input shape differs')
  bit_invalid=(data.view(np.uint32) & np.uint32(0x7f800000))==np.uint32(0x7f800000)
  if not np.array_equal(bit_invalid,~np.isfinite(data)):raise RuntimeError('bit classification differs from CPU finite classification')
  if 'invalid_word' in c and int(data.view(np.uint32)[c['invalid_index']])!=c['invalid_word']:raise RuntimeError('NaN payload differs')
  invalid=np.flatnonzero(~np.isfinite(data));first=int(invalid[0]) if len(invalid) else data.size
  if (first if len(invalid) else -1)!=c.get('invalid_index',-1):raise RuntimeError('nonfinite fixture location differs')
  if 'invalid_locations' in c and invalid.tolist()!=c['invalid_locations']:raise RuntimeError('multiple invalid fixture positions differ')
  expected=data.copy();prefix=(data[:first]+np.float32(1))*np.float32(.5);expected[:first]=np.minimum(np.float32(1),np.maximum(np.float32(0),prefix))
  clamped=folder/('case%02d.clamped.f32'%c['index'])
  if clamped.read_bytes()!=expected.tobytes():raise RuntimeError('complete CPU clamp bytes differ: '+c['kind'])
  cpu_float+=1;files.extend([source,clamped])
  if 'source' in c and sha(ROOT/c['source'])!=c['source_sha256']:raise RuntimeError('real decoded changed')
  if not len(invalid):
   scaled=(expected.reshape(3,c['spatial'])*np.float32(255)).astype(np.float64)
   packed=np.floor(scaled+.5).astype(np.uint8).T.copy().tobytes();rgb=folder/('case%02d.rgb'%c['index'])
   if rgb.read_bytes()!=packed:raise RuntimeError('complete CPU rounded RGB bytes differ: '+c['kind'])
   cpu_rgb+=1;files.append(rgb)
 records=re.findall(r'PIXEL_TIME index=(\d+) arm=(\d+) rep=(\d+) kernel=(reference|candidate) clamp_seconds=([0-9.]+) pack_seconds=([0-9.]+) full_bytes=(\d+)',text)
 order=[(c['index'],arm,rep) for c in cases if 'invalid_index' not in c for arm in range(4) for rep in range(4)]
 if len(records)!=320:raise RuntimeError('320 ordered timed pairs required')
 times=[]
 for r,key in zip(records,order):
  if tuple(map(int,r[:3]))!=key or r[3]!=('candidate' if key[1] in (1,2) else 'reference') or r[6]!='1' or any(not math.isfinite(float(t)) or float(t)<=0 for t in r[4:6]):raise RuntimeError('timing order/mode/value differs')
  times.append(dict(index=key[0],arm=key[1],rep=key[2],clamp_seconds=float(r[4]),pack_seconds=float(r[5])))
 comparisons=[]
 for c in cases:
  if 'invalid_index' in c:continue
  item=dict(index=c['index'],kind=c['kind'],spatial=c['spatial'])
  for part in ('clamp','pack'):
   values={mode:[r[part+'_seconds'] for r in times if r['index']==c['index'] and r['arm'] in arms and r['rep']>0] for mode,arms in (('reference',(0,3)),('candidate',(1,2)))}
   b=sum(values['reference'])/6;t=sum(values['candidate'])/6
   item[part]=dict(reference_samples_seconds=values['reference'],candidate_samples_seconds=values['candidate'],reference_mean_seconds=b,candidate_mean_seconds=t,latency_reduction_percent=(1-t/b)*100)
  comparisons.append(item)
 real=[r for r in comparisons if r['kind'].startswith('cpu_real_')];real_mean={}
 if len(real)!=6:raise RuntimeError('six real CPU decoded fixtures required')
 for part in ('clamp','pack'):
  b=sum(r[part]['reference_mean_seconds'] for r in real)/6;t=sum(r[part]['candidate_mean_seconds'] for r in real)/6;real_mean[part]=dict(reference_seconds=b,candidate_seconds=t,latency_reduction_percent=(1-t/b)*100)
 temp=ROOT/re.findall(r'Temperature guard:.*log=([^\s]+)',text)[-1];fans=ROOT/re.findall(r'Fan observation:.*log=([^\s]+)',text)[-1];rows=list(csv.DictReader(temp.open()))
 if not rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in rows):raise RuntimeError('unsafe thermal evidence')
 files.extend([temp,fans]+[ROOT/n for n in ('src/ve_sd_turbo_pixels_bits.cpp','src/ve_sd_turbo_pixels_pack.cpp','tests/benchmark_sd_pixels_pack.cpp','scripts/benchmark_sd_pixels_pack.sh','scripts/prepare_sd_pixels_pack_fixtures.py','build/sd-pixels-pack-probe/reference.o','build/sd-pixels-pack-probe/candidate.o','build/sd-pixels-pack-probe/probe','build/sd-pixels-pack-probe/direct','tests/check_sd_pixels_pack_direct.cpp')])
 prior_path=ROOT/'docs/results/20261009T171542Z-sd-turbo-pixels-bits.json';prior=json.loads(prior_path.read_text())
 if sha(ROOT/'src/ve_sd_turbo_pixels_bits.cpp')!=prior['artifact_sha256']['src/ve_sd_turbo_pixels_bits.cpp']:raise RuntimeError('baseline source differs from accepted O2')
 baseline_fixture=ROOT/'build/sd-pixels-bits-probe/fixtures/manifest.json'
 if sha(baseline_fixture)!=prior['artifact_sha256'][str(baseline_fixture.relative_to(ROOT))]:raise RuntimeError('accepted baseline fixtures changed')
 baseline_cases=json.loads(baseline_fixture.read_text())['cases']
 if [c['input_sha256'] for c in cases]!=[c['input_sha256'] for c in baseline_cases]:raise RuntimeError('same expanded fixtures required')
 files.append(baseline_fixture)
 diagnostics=[dict(source=source,line=int(line),message=message) for source,line,message in re.findall(r'^nc\+\+: (?:vec|opt)\([^)]*\): (src/ve_sd_turbo_pixels(?:_bits|_pack)?\.cpp), line (\d+): ([^\n]+)',text,re.M)]
 if not any(r['source']=='src/ve_sd_turbo_pixels_pack.cpp' and r['line']==33 and r['message']=='Vectorized loop.' for r in diagnostics):raise RuntimeError('actual candidate clamp vectorization evidence required')
 if not any(r['source']=='src/ve_sd_turbo_pixels_pack.cpp' and r['line']==42 and r['message']=='Partially vectorized loop.' for r in diagnostics):raise RuntimeError('actual pack compilation evidence required')
 if not any(r['source']=='src/ve_sd_turbo_pixels_pack.cpp' and r['line']==26 and r['message']=='Partially vectorized loop.' for r in diagnostics):raise RuntimeError('actual finite scan partial vectorization required')
 symbol_path=ROOT/'build/sd-pixels-pack-symbols.log';reloc_path=ROOT/'build/sd-pixels-pack-relocations.log'
 if 'Symbol table' not in symbol_path.read_text() or 'Relocation section' not in reloc_path.read_text():raise RuntimeError('complete object inspection required')
 if any(word in symbol_path.read_text()+reloc_path.read_text() for word in ('isfinite','__libcpp_isfinite','round')):raise RuntimeError('finite wrapper remains in candidate object')
 files.extend([prior_path,symbol_path,reloc_path])
 direct_comparisons,direct_files=audit_direct_pack(folder,text)
 files.extend(direct_files)
 output=dict(status='pixel_pack_microbenchmark_verified',model_speedup_measured=False,compiler_diagnostics=diagnostics,clamp_compute_vectorized=True,finite_scan_vectorized=any(r['source']=='src/ve_sd_turbo_pixels_pack.cpp' and r['line']==26 and r['message']=='Vectorized loop.' for r in diagnostics),finite_scan_partially_vectorized=True,finite_wrapper_present=False,pack_vectorized=False,pack_partially_vectorized=True,round_wrapper_present=False,clamp_is_control=True,accepted_baseline_proof_sha256=sha(prior_path),comparisons=comparisons,real_six_mean=real_mean,full_cpu_float_checks=cpu_float,full_cpu_rgb_checks=cpu_rgb+12,pipeline_cpu_rgb_checks=cpu_rgb,direct_cpu_rgb_checks=12,direct_comparisons=direct_comparisons,direct_native_timed_calls=192,half_threshold_groups=255,native_timed_pairs=320,native_invalid_cases=23,native_invalid_argument_checks=7,all_native_float_and_rgb_bytes_identical=True,temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},publisher_sha256=sha(Path(__file__)),compiler_flags=dict(reference='-O2 -std=c++11 -DNDEBUG -fno-fast-math -fno-associative-math -fdiag-vector=2 -Dve_sd_turbo_pixel_clamp=sd_pixels_reference_clamp -Dve_sd_turbo_pixel_pack=sd_pixels_reference_pack',candidate='-O2 -std=c++11 -DNDEBUG -fno-fast-math -fno-associative-math -fdiag-vector=2'),scope='single VE independent exact integer-threshold O2 RGB packing; identical bit-pattern clamp is a control, not an optimization, 20 finite including 6 original CPU decoded outputs and 23 nonfinite fixtures plus 12 direct finite [0,1] pack fixtures, 255 half-threshold groups with five neighboring values and 192 direct timings; ABBA one warmup three timings per arm; full float/RGB CPU byte checks; no full-model speedup claim')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('Pixel suite audited:',target.relative_to(ROOT));print(real_mean)
if __name__=='__main__':main()
