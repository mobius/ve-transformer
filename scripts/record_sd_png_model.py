import argparse
import json
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe
from record_sd_gemm_spatial_model import audited_run
ROOT=Path(__file__).resolve().parents[1]
def png_dispatch(folder,value,mode):
 raw=(folder/'native.log').read_text()
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S)
 if len(bodies)!=len(value['requests']) or value.get('png_encoder')!=mode or value.get('image_output_profile_enabled'):
  raise RuntimeError('PNG mode or diagnostic configuration differs')
 for body,request in zip(bodies,value['requests']):
  rows=re.findall(r'SD_PNG_ENCODER mode=(generic|ve) width=(\d+) height=(\d+) channels=(\d+) stride=(\d+)',body)
  expected=dict(mode=mode,width=512,height=512,channels=3,stride=1536)
  if rows!=[(mode,'512','512','3','1536')] or body.count('SD_PNG_ENCODER ')!=1 or request.get('png_encoder_dispatch')!=expected:
   raise RuntimeError('actual encoder dispatch differs')
def png_compiler_evidence():
 folder=ROOT/'build/sd-baseline-ve/CMakeFiles/sd-png-ve.dir'
 flags_path=folder/'flags.make';make_path=folder/'build.make'
 flags=re.findall(r'^CXX_FLAGS = (.*)$',flags_path.read_text(),re.M)
 commands=[line for line in make_path.read_text().splitlines() if '$(CXX_FLAGS)' in line and ' -c ' in line and 've_sd_turbo_png.cpp' in line]
 objects=list(folder.rglob('ve_sd_turbo_png.cpp.o'))
 if len(flags)!=1 or len(commands)!=1 or len(objects)!=1:raise RuntimeError('unique actual PNG compilation required')
 levels=re.findall(r'(?<!\S)-O[0-3sg](?!\S)',flags[0]+' '+commands[0])
 main_flags_path=ROOT/'build/sd-baseline-ve/examples/cli/CMakeFiles/sd.dir/flags.make'
 main_flags=re.findall(r'^CXX_FLAGS = (.*)$',main_flags_path.read_text(),re.M)
 if not levels or levels[-1]!='-O2' or len(main_flags)!=1 or re.findall(r'(?<!\S)-O[0-3sg](?!\S)',main_flags[0])[-1]!='-O0':raise RuntimeError('isolated PNG O2 and original main O0 required')
 return dict(effective_png_optimization=levels[-1],png_flags=flags[0],png_object_sha256=sha(objects[0]),png_flags_make_sha256=sha(flags_path),png_build_make_sha256=sha(make_path),main_flags=main_flags[0],main_optimization='-O0',main_flags_make_sha256=sha(main_flags_path))
def audited_png_run(run,memory,guard,mode):
 value,entry,manifest=audited_run(run,memory,guard,'8192');png_dispatch(run,value,mode)
 return value,entry,manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results')
 paths=[a.run.resolve(),a.memory.resolve(),a.guard_log.resolve()]
 for path in paths:path.relative_to(ROOT/'build')
 result,entry,manifest=audited_png_run(*paths,'ve')
 compiler=png_compiler_evidence()
 steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[a.name]
 if result['steps']!=steps or [r['reference_case'] for r in result['requests']]!=cases:raise RuntimeError('wrong workload')
 value=json.loads(proof.read_text()) if proof.exists() else dict(status='model_validation_in_progress',tests=[],model_manifest_sha256=manifest,binary_sha256=result['binary_sha256'],checker_sha256=result['checker_sha256'])
 if value['model_manifest_sha256']!=manifest or value['binary_sha256']!=result['binary_sha256'] or value['checker_sha256']!=result['checker_sha256']:raise RuntimeError('candidate build/checker changed')
 existing=[e for e in value['tests'] if e['name']!=a.name]
 for e in existing:
  for name,digest in e['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('earlier test evidence changed')
 if value.get('actual_png_compiler',compiler)!=compiler:raise RuntimeError('actual encoder object or flags changed')
 value['actual_png_compiler']=compiler
 entry['name']=a.name;value['tests']=sorted(existing+[entry],key=lambda e:('double','six','four').index(e['name']))
 value['completed_tests']=[e['name'] for e in value['tests']];value['independent_cpu_checks']=sum(e['cpu_checks'] for e in value['tests']);value['all_candidate_tests_completed']=value['completed_tests']==['double','six','four'];value['status']='model_validation_verified' if value['all_candidate_tests_completed'] else 'model_validation_in_progress'
 value['publisher_sha256']=sha(Path(__file__));value['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')};value['full_graph_speedup_measured']=False;value['configuration']=dict(png_encoder='ve',vae_spatial_tile='8192',im2col_mode='rows_256',generic_threads=8,vae_blas_threads=4,gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');value['scope']='isolated actual-model O2 PNG candidate correctness, original main O0, no old-build CPU checks; controlled same-binary performance separate'
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Model tests audited:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])
if __name__=='__main__':main()
