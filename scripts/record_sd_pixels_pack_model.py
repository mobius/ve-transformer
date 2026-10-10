import argparse
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe
from record_sd_rgb_model import audited_rgb_run
from record_sd_png_model import png_compiler_evidence
ROOT=Path(__file__).resolve().parents[1]
def pixels_compiler_evidence():
 folder=ROOT/'build/sd-baseline-ve/CMakeFiles/sd-pixels-ve.dir'
 flags_path=folder/'flags.make';make_path=folder/'build.make';archive=ROOT/'build/sd-baseline-ve/libsd-pixels-ve.a';link=ROOT/'build/sd-baseline-ve/examples/cli/CMakeFiles/sd.dir/link.txt'
 flags=re.findall(r'^CXX_FLAGS = (.*)$',flags_path.read_text(),re.M)
 commands=[line for line in make_path.read_text().splitlines() if '$(CXX_FLAGS)' in line and ' -c ' in line and 've_sd_turbo_pixels_pack.cpp' in line]
 objects=list(folder.rglob('ve_sd_turbo_pixels_pack.cpp.o'))
 if len(flags)!=1 or len(commands)!=1 or len(objects)!=1:raise RuntimeError('unique actual pack compiler required')
 levels=re.findall(r'(?<!\S)-O[0-3sg](?!\S)',flags[0]+' '+commands[0])
 if not levels or levels[-1]!='-O2' or '-fno-fast-math' not in flags[0] or '-fno-associative-math' not in flags[0]:raise RuntimeError('strict actual pack O2 required')
 if subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()!=['ve_sd_turbo_pixels_pack.cpp.o'] or 'libsd-pixels-ve.a' not in link.read_text():raise RuntimeError('actual pixel archive/link differs')
 objsymbols=subprocess.check_output(['readelf','-Ws',str(objects[0])],text=True);binsymbols=subprocess.check_output(['readelf','-Ws',str(ROOT/'build/sd-baseline-ve/bin/sd')],text=True)
 if any(word in objsymbols for word in ('isfinite','__libcpp_isfinite','memcpy','round')):raise RuntimeError('finite classification wrapper remains')
 sizes={}
 for name in ('ve_sd_turbo_pixel_clamp','ve_sd_turbo_pixel_pack'):
  pattern=r'^\s*\d+:\s+\S+\s+(\d+)\s+FUNC\s+GLOBAL\s+DEFAULT\s+\S+\s+'+name+r'$'
  obj=re.findall(pattern,objsymbols,re.M);binary=re.findall(pattern,binsymbols,re.M)
  if len(obj)!=1 or len(binary)!=1 or obj!=binary:raise RuntimeError('actual linked pixel symbol sizes differ')
  sizes[name]=int(obj[0])
 build_log=ROOT/'build/sd-pixels-pack-model-build.log';raw=build_log.read_text()
 diagnostics=[dict(line=int(line),message=message) for line,message in re.findall(r've_sd_turbo_pixels_pack\.cpp, line (\d+): ([^\n]+)',raw)]
 if not any(r==dict(line=26,message='Partially vectorized loop.') for r in diagnostics) or not any(r==dict(line=33,message='Vectorized loop.') for r in diagnostics):raise RuntimeError('actual model compilation vectorization evidence required')
 if not any(r==dict(line=42,message='Partially vectorized loop.') for r in diagnostics):raise RuntimeError('actual model pack partial vectorization required')
 prior_path=ROOT/'docs/results/20261009T175942Z-sd-turbo-pixels-pack.json';prior=json.loads(prior_path.read_text())
 if prior['status']!='pixel_pack_microbenchmark_verified' or prior['full_cpu_float_checks']!=43 or prior['full_cpu_rgb_checks']!=32 or sha(ROOT/'src/ve_sd_turbo_pixels_pack.cpp')!=prior['artifact_sha256']['src/ve_sd_turbo_pixels_pack.cpp']:raise RuntimeError('actual source differs from tested independent candidate')
 return dict(effective_pixels_optimization=levels[-1],pixels_flags=flags[0],pixels_object_sha256=sha(objects[0]),pixels_flags_make_sha256=sha(flags_path),pixels_build_make_sha256=sha(make_path),pixels_source_sha256=sha(ROOT/'src/ve_sd_turbo_pixels_pack.cpp'),pixels_archive_sha256=sha(archive),main_link_sha256=sha(link),linked_symbol_sizes=sizes,compiler_diagnostics=diagnostics,build_log_sha256=sha(build_log),independent_kernel_proof_sha256=sha(prior_path))


def recompute_cpu_reference(run,value):
 reference=ROOT/value['reference_artifacts'];cpu=json.loads((reference/'summary.json').read_text())
 if not cpu['completed'] or cpu['size']!=512 or cpu['steps']!=value['steps']:raise RuntimeError('completed matching CPU reference required')
 rows={row['case']:row for row in cpu['cases']};checks=[];files=[reference/'summary.json'];png_count=0
 for request in value['requests']:
  row=rows[request['reference_case']];source=reference/('case%d'%request['reference_case']);output=run/('request%d'%request['request'])
  for name,item in row['arrays'].items():
   path=source/(name+'.f32')
   if sha(path)!=item['sha256']:raise RuntimeError('CPU reference array changed')
   files.append(path)
  def array(name):return np.fromfile(source/(name+'.f32'),dtype=np.float32).reshape(row['arrays'][name]['shape'])
  expected=[('native-embeddings',array('embeddings'),1e-4)]
  for step in range(value['steps']):
   expected.extend([('native-step%d-input'%step,array('step%d-input'%step),1e-4),('native-step%d-epsilon'%step,array('step%d-epsilon'%step),1e-3)])
   timestep=np.fromfile(output/('native-step%d-timestep.f32'%step),dtype=np.float32)
   if timestep.size!=1 or abs(float(timestep[0])-float(array('timesteps').reshape(-1)[step]))>1e-3:raise RuntimeError('actual timestep differs')
  expected.extend([('native-latent',array('step%d-latent'%(value['steps']-1)),1e-2),('native-decoded',np.clip((array('decoded')+np.float32(1))*np.float32(.5),0,1),1e-2)])
  if [r['name'] for r in request['checks']]!=[name for name,_,_ in expected] or sha(output/'native-noise.f32')!=row['arrays']['noise']['sha256']:raise RuntimeError('actual check order or noise differs')
  for (name,target,tolerance),record in zip(expected,request['checks']):
   actual=np.fromfile(output/(name+'.f32'),dtype=np.float32);target=np.asarray(target,dtype=np.float32).reshape(-1)
   if actual.shape!=target.shape or not np.isfinite(actual).all() or not np.isfinite(target).all():raise RuntimeError('CPU audit shape/finiteness differs')
   delta=actual.astype(np.float64)-target.astype(np.float64);relative=float(np.linalg.norm(delta)/max(float(np.linalg.norm(target.astype(np.float64))),1e-12));maximum=float(np.max(np.abs(delta)))
   if relative>tolerance or record['tolerance']!=tolerance or not np.isclose(relative,record['relative_l2'],rtol=1e-12,atol=1e-15) or maximum!=record['max_abs_error']:raise RuntimeError('recomputed CPU error differs or exceeds tolerance')
   checks.append(dict(request=request['request'],name=name,relative_l2=relative,max_abs_error=maximum,tolerance=tolerance,passed=True))
  decoded=np.fromfile(output/'native-decoded.f32',dtype=np.float32).reshape(3,512*512)
  if not ((decoded>=0)&(decoded<=1)).all():raise RuntimeError('actual packing input domain differs')
  packed=np.floor((decoded*np.float32(255)).astype(np.float64)+.5).astype(np.uint8).T.copy().tobytes()
  with Image.open(output/'image.png') as image:
   if image.size!=(512,512) or image.mode!='RGB' or image.tobytes()!=packed:raise RuntimeError('PNG full decoded RGB bytes differ')
  png_count+=1
 return checks,png_count,{str(p.relative_to(ROOT)):sha(p) for p in set(files)}

def audited_pixels_run(run,memory,guard,mode):
 value,entry,manifest=audited_rgb_run(run,memory,guard,'resident')
 raw=(run/'native.log').read_text()
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S)
 if value.get('pixel_kernel')!=mode or len(bodies)!=len(value['requests']):raise RuntimeError('pixel configuration differs')
 for body,request in zip(bodies,value['requests']):
  rows=re.findall(r'SD_PIXEL_KERNEL mode=(generic|ve) clamp_calls=(\d+) pack_calls=(\d+) spatial=(\d+) bytes=(\d+)',body)
  expected=dict(mode=mode,clamp_calls=1,pack_calls=1,spatial=262144,bytes=786432)
  if rows!=[(mode,'1','1','262144','786432')] or body.count('SD_PIXEL_KERNEL ')!=1 or request.get('pixel_kernel_dispatch')!=expected:raise RuntimeError('actual pixel dispatch differs')
 entry['pixel_kernel']=mode
 return value,entry,manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results')
 paths=[a.run.resolve(),a.memory.resolve(),a.guard_log.resolve()]
 for path in paths:path.relative_to(ROOT/'build')
 result,entry,manifest=audited_pixels_run(*paths,'ve')
 recomputed,png_count,reference_sha=recompute_cpu_reference(paths[0],result)
 if len(recomputed)!=entry['cpu_checks']:raise RuntimeError('recomputed CPU check count differs')
 entry['cpu_recomputed_checks']=recomputed;entry['cpu_recomputed_png_checks']=png_count;entry['cpu_reference_sha256']=reference_sha;entry['packing_input_domain_verified']=True
 compiler=png_compiler_evidence();pixels_compiler=pixels_compiler_evidence()
 steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[a.name]
 if result['steps']!=steps or [r['reference_case'] for r in result['requests']]!=cases:raise RuntimeError('wrong workload')
 value=json.loads(proof.read_text()) if proof.exists() else dict(status='model_validation_in_progress',tests=[],model_manifest_sha256=manifest,binary_sha256=result['binary_sha256'],checker_sha256=result['checker_sha256'])
 if value['model_manifest_sha256']!=manifest or value['binary_sha256']!=result['binary_sha256'] or value['checker_sha256']!=result['checker_sha256']:raise RuntimeError('candidate build/checker changed')
 existing=[e for e in value['tests'] if e['name']!=a.name]
 for e in existing:
  for name,digest in e['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('earlier test evidence changed')
 if value.get('actual_png_compiler',compiler)!=compiler:raise RuntimeError('actual encoder object or flags changed')
 if value.get('actual_pixels_compiler',pixels_compiler)!=pixels_compiler:raise RuntimeError('actual pixels compiler changed')
 value['actual_pixels_compiler']=pixels_compiler
 value['actual_png_compiler']=compiler
 entry['name']=a.name;value['tests']=sorted(existing+[entry],key=lambda e:('double','six','four').index(e['name']))
 value['completed_tests']=[e['name'] for e in value['tests']];value['independent_cpu_checks']=sum(e['cpu_checks'] for e in value['tests']);value['all_candidate_tests_completed']=value['completed_tests']==['double','six','four'];value['status']='model_validation_verified' if value['all_candidate_tests_completed'] else 'model_validation_in_progress'
 value['cpu_recomputed_checks']=sum(len(e['cpu_recomputed_checks']) for e in value['tests']);value['cpu_recomputed_png_checks']=sum(e['cpu_recomputed_png_checks'] for e in value['tests'])
 value['publisher_sha256']=sha(Path(__file__));value['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_rgb_model.py','scripts/record_sd_png_model.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')};value['full_graph_speedup_measured']=False;value['configuration']=dict(pixel_kernel='ve',rgb_buffer='resident',png_encoder='ve',vae_spatial_tile='8192',im2col_mode='rows_256',generic_threads=8,vae_blas_threads=4,gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');value['scope']='actual-model exact RGB O2 packing correctness and independent recomputed CPU errors/full PNG RGB bytes; PNG O2/main O0; incremental full graph speedup requires a controlled comparison'
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Model tests audited:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])
if __name__=='__main__':main()
