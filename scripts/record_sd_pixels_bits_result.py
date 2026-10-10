"""Audit full CPU float/RGB outputs and raw VE pixel ABBA timings."""
import argparse,csv,json,math,re
from pathlib import Path
import numpy as np
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();log=a.log.resolve();log.relative_to(ROOT/'build');text=log.read_text()
 artifacts=re.findall(r'Pixel artifacts: (build/results/[^\s]+)',text)
 if len(artifacts)!=1 or text.count('PIXEL_COMPLETE cases=43 finite=20 timed_pairs=320 invalid=23')!=1 or text.count('PIXEL_INVALID_ARGUMENTS checks=7 PASS')!=1 or 'Temperature guard stopped command' in text:raise RuntimeError('complete native pixel suite required')
 folder=ROOT/artifacts[0];manifest_path=ROOT/'build/sd-pixels-bits-probe/fixtures/manifest.json';manifest=json.loads(manifest_path.read_text());cases=manifest['cases']
 if len(cases)!=43 or [c['index'] for c in cases]!=list(range(43)) or sha(ROOT/manifest['reference_summary'])!=manifest['reference_summary_sha256'] or sha(ROOT/'scripts/prepare_sd_pixels_bits_fixtures.py')!=manifest['generator_sha256']:raise RuntimeError('fixture provenance differs')
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
 files.extend([temp,fans]+[ROOT/n for n in ('src/ve_sd_turbo_pixels_vector.cpp','src/ve_sd_turbo_pixels_bits.cpp','tests/benchmark_sd_pixels_bits.cpp','scripts/benchmark_sd_pixels_bits.sh','scripts/prepare_sd_pixels_bits_fixtures.py','build/sd-pixels-bits-probe/reference.o','build/sd-pixels-bits-probe/candidate.o','build/sd-pixels-bits-probe/probe')])
 prior_path=ROOT/'docs/results/20261009T165801Z-sd-turbo-pixels-vector.json';prior=json.loads(prior_path.read_text())
 if sha(ROOT/'src/ve_sd_turbo_pixels_vector.cpp')!=prior['artifact_sha256']['src/ve_sd_turbo_pixels_vector.cpp']:raise RuntimeError('baseline source differs from accepted O2')
 baseline_fixture=ROOT/'build/sd-pixels-vector-probe/fixtures/manifest.json'
 if sha(baseline_fixture)!=prior['artifact_sha256'][str(baseline_fixture.relative_to(ROOT))]:raise RuntimeError('accepted baseline fixtures changed')
 baseline_cases=json.loads(baseline_fixture.read_text())['cases']
 if [c['input_sha256'] for c in cases[:33]]!=[c['input_sha256'] for c in baseline_cases]:raise RuntimeError('same expanded fixtures required')
 files.append(baseline_fixture)
 diagnostics=[dict(source=source,line=int(line),message=message) for source,line,message in re.findall(r'^nc\+\+: (?:vec|opt)\([^)]*\): (src/ve_sd_turbo_pixels(?:_vector|_bits)?\.cpp), line (\d+): ([^\n]+)',text,re.M)]
 if not any(r['source']=='src/ve_sd_turbo_pixels_bits.cpp' and r['line']==33 and r['message']=='Vectorized loop.' for r in diagnostics):raise RuntimeError('actual candidate clamp vectorization evidence required')
 if not any(r['source']=='src/ve_sd_turbo_pixels_bits.cpp' and r['line']==40 and r['message']=='Unvectorized loop.' for r in diagnostics):raise RuntimeError('actual pack compilation evidence required')
 if not any(r['source']=='src/ve_sd_turbo_pixels_bits.cpp' and r['line']==26 and r['message']=='Partially vectorized loop.' for r in diagnostics):raise RuntimeError('actual finite scan partial vectorization required')
 symbol_path=ROOT/'build/sd-pixels-bits-symbols.log';reloc_path=ROOT/'build/sd-pixels-bits-relocations.log'
 if 'Symbol table' not in symbol_path.read_text() or 'Relocation section' not in reloc_path.read_text():raise RuntimeError('complete object inspection required')
 if any(word in symbol_path.read_text()+reloc_path.read_text() for word in ('isfinite','__libcpp_isfinite')):raise RuntimeError('finite wrapper remains in candidate object')
 files.extend([prior_path,symbol_path,reloc_path])
 output=dict(status='pixel_bits_microbenchmark_verified',model_speedup_measured=False,compiler_diagnostics=diagnostics,clamp_compute_vectorized=True,finite_scan_vectorized=any(r['source']=='src/ve_sd_turbo_pixels_bits.cpp' and r['line']==26 and r['message']=='Vectorized loop.' for r in diagnostics),finite_scan_partially_vectorized=True,finite_wrapper_present=False,pack_vectorized=False,accepted_baseline_proof_sha256=sha(prior_path),comparisons=comparisons,real_six_mean=real_mean,full_cpu_float_checks=cpu_float,full_cpu_rgb_checks=cpu_rgb,native_timed_pairs=320,native_invalid_cases=23,native_invalid_argument_checks=7,all_native_float_and_rgb_bytes_identical=True,temperature=thermal(log),fan_detail=fan_detail(log),artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},publisher_sha256=sha(Path(__file__)),compiler_flags=dict(reference='-O2 -std=c++11 -DNDEBUG -fno-fast-math -fno-associative-math -fdiag-vector=2 -Dve_sd_turbo_pixel_clamp=sd_pixels_reference_clamp -Dve_sd_turbo_pixel_pack=sd_pixels_reference_pack',candidate='-O2 -std=c++11 -DNDEBUG -fno-fast-math -fno-associative-math -fdiag-vector=2'),scope='single VE independent bit-pattern/any-invalid scan O2 pixel kernels; identical packing is a control, not an optimization, 20 finite including 6 original CPU decoded outputs and 23 nonfinite fixtures; ABBA one warmup three timings per arm; full float/RGB CPU byte checks; no full-model speedup claim')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('Pixel suite audited:',target.relative_to(ROOT));print(real_mean)
if __name__=='__main__':main()
