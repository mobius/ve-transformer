"""Audit actual-model selective CONT dispatch, exact object and CPU/PNG outputs."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_model import audited_pixels_run,recompute_cpu_reference
from record_sd_png_model import png_compiler_evidence
from record_qwen36_mtp import safe
ROOT=Path(__file__).resolve().parents[1]
def cont_dispatch(run,value,mode):
 raw=(run/'native.log').read_text();bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S)
 if len(bodies)!=len(value['requests']) or 'SD_CONT_SHAPE ' in raw:raise RuntimeError('unprofiled ordered CONT requests required')
 parsed=[]
 for body in bodies:
  rows=re.findall(r'SD_CONT_DISPATCH stage=(clip|unet|vae) optimized=(\d+) fallback=(\d+) enabled=(0|1)',body)
  if len(rows)!=body.count('SD_CONT_DISPATCH '):raise RuntimeError('complete CONT markers required')
  entries=[dict(stage=s,optimized=int(o),fallback=int(f),enabled=int(e)) for s,o,f,e in rows]
  for stage,total in (('clip',92),('unet',258),('vae',43)):
   actual=[r for r in entries if r['stage']==stage];optimized=15 if mode=='ve' and stage=='unet' else 0
   expected=dict(stage=stage,optimized=optimized,fallback=total-optimized,enabled=int(mode=='ve'))
   if len(actual)!=(value['steps'] if stage=='unet' else 1) or any(r!=expected for r in actual):raise RuntimeError('actual CONT candidate/fallback count differs')
  if [r['stage'] for r in entries]!=['clip']+['unet']*value['steps']+['vae']:raise RuntimeError('actual CONT stage ordering differs')
  parsed.append(entries)
 return parsed

def compiler_evidence():
 folder=ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir';flags_path=folder/'flags.make';make_path=folder/'build.make';flags=re.findall(r'^C_FLAGS = (.*)$',flags_path.read_text(),re.M)
 commands=[r for r in make_path.read_text().splitlines() if '$(C_FLAGS)' in r and ' -c ' in r and 've_sd_turbo_cont_transpose.c' in r];objects=list(folder.rglob('ve_sd_turbo_cont_transpose.c.o'))
 if len(flags)!=1 or len(commands)!=1 or len(objects)!=1:raise RuntimeError('unique actual CONT object/command required')
 effective=flags[0]+' '+commands[0];levels=re.findall(r'(?<!\S)-O[0-3sg](?!\S)',effective)
 if not levels or levels[-1]!='-O2' or '-fno-fast-math' not in effective or '-fno-associative-math' not in effective:raise RuntimeError('actual strict CONT O2 required')
 proof=ROOT/'docs/results/20261009T184817Z-sd-turbo-cont-transpose.json';prior=json.loads(proof.read_text());source=ROOT/'src/ve_sd_turbo_cont_transpose.c'
 if prior['status']!='cont_transpose_microbenchmark_verified' or prior['cpu_bit_checks']!=6 or prior['checks']!={'ownership':180,'concurrent':48,'graph':24,'invalid':14} or sha(source)!=prior['artifact_sha256']['src/ve_sd_turbo_cont_transpose.c']:raise RuntimeError('tested independent CONT source required')
 if sha(objects[0])!=prior['artifact_sha256']['build/sd-cont-transpose-probe/candidate.o']:raise RuntimeError('actual CONT object differs from independently tested object')
 obj=subprocess.check_output(['readelf','-Ws',str(objects[0])],text=True);binary=subprocess.check_output(['readelf','-Ws',str(ROOT/'build/sd-baseline-ve/bin/sd')],text=True)
 pattern=r'^\s*\d+:\s+\S+\s+(\d+)\s+FUNC\s+GLOBAL\s+DEFAULT\s+\S+\s+sd_ve_cont_transpose_f32$';sizes=re.findall(pattern,obj,re.M)
 if len(sizes)!=1 or re.findall(pattern,binary,re.M)!=sizes:raise RuntimeError('actual linked CONT symbol differs')
 build=ROOT/'build/sd-cont-model-build.log';object_build=ROOT/'build/sd-cont-model-object-build.log';diagnostics=[dict(line=int(n),message=s.strip()) for n,s in re.findall(r've_sd_turbo_cont_transpose\.c, line (\d+): ([^\n]+)',object_build.read_text())]
 if dict(line=10,message='Vectorized loop.') not in diagnostics:raise RuntimeError('actual model vectorized copy evidence required')
 return dict(effective_optimization='-O2',compiler_flags=flags[0],object_sha256=sha(objects[0]),source_sha256=sha(source),flags_make_sha256=sha(flags_path),build_make_sha256=sha(make_path),linked_function_size=int(sizes[0]),compiler_diagnostics=diagnostics,build_log_sha256=sha(build),object_compile_log_sha256=sha(object_build),independent_proof_sha256=sha(proof),actual_object_matches_independent=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args();proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');paths=[x.resolve() for x in (a.run,a.memory,a.guard_log)]
 for path in paths:path.relative_to(ROOT/'build')
 result,entry,manifest=audited_pixels_run(*paths,'ve');dispatch=cont_dispatch(paths[0],result,'ve');recomputed,png_count,reference_sha=recompute_cpu_reference(paths[0],result);compiler=compiler_evidence();png=png_compiler_evidence()
 steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[a.name]
 if result['steps']!=steps or [r['reference_case'] for r in result['requests']]!=cases or len(recomputed)!=entry['cpu_checks']:raise RuntimeError('matching complete model workload required')
 entry.update(name=a.name,cont_kernel='ve',cont_dispatch=dispatch,cpu_recomputed_checks=recomputed,cpu_recomputed_png_checks=png_count,cpu_reference_sha256=reference_sha,packing_input_domain_verified=True)
 v=json.loads(proof.read_text()) if proof.exists() else dict(status='model_validation_in_progress',tests=[],model_manifest_sha256=manifest,binary_sha256=result['binary_sha256'],checker_sha256=result['checker_sha256'])
 if v['model_manifest_sha256']!=manifest or v['binary_sha256']!=result['binary_sha256'] or v['checker_sha256']!=result['checker_sha256'] or v.get('actual_cont_compiler',compiler)!=compiler or v.get('actual_png_compiler',png)!=png:raise RuntimeError('candidate model/objects/checker changed')
 old=[e for e in v['tests'] if e['name']!=a.name]
 for e in old:
  for name,h in e['sha256'].items():
   if sha(ROOT/name)!=h:raise RuntimeError('previous test evidence changed')
 v['tests']=sorted(old+[entry],key=lambda e:('double','six','four').index(e['name']));v['completed_tests']=[e['name'] for e in v['tests']];v['independent_cpu_checks']=sum(e['cpu_checks'] for e in v['tests']);v['cpu_recomputed_checks']=sum(len(e['cpu_recomputed_checks']) for e in v['tests']);v['cpu_recomputed_png_checks']=sum(e['cpu_recomputed_png_checks'] for e in v['tests']);v['all_candidate_tests_completed']=v['completed_tests']==['double','six','four'];v['status']='model_validation_verified' if v['all_candidate_tests_completed'] else 'model_validation_in_progress';v['actual_cont_compiler']=compiler;v['actual_png_compiler']=png;v['publisher_sha256']=sha(Path(__file__));v['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_pixels_pack_model.py','scripts/record_sd_rgb_model.py','scripts/record_sd_png_model.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')};v['full_graph_speedup_measured']=False;v['configuration']=dict(cont_kernel='ve',pixel_kernel='ve',rgb_buffer='resident',png_encoder='ve',vae_spatial_tile='8192',im2col_mode='rows_256',generic_threads=8,vae_blas_threads=4,gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');v['scope']='actual-model selective CONT transpose O2 correctness and independently recomputed CPU/PNG; actual object equal independently tested object; 15 UNet nodes/step; full request incremental speedup requires controlled comparison'
 safe(v);proof.write_text(json.dumps(v,indent=2)+'\n');print('CONT model tests audited:',v['completed_tests'],'CPU checks:',v['independent_cpu_checks'])
if __name__=='__main__':main()
