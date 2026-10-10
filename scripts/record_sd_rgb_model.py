import argparse
import json
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe
from record_sd_png_model import audited_png_run,png_compiler_evidence
ROOT=Path(__file__).resolve().parents[1]
def audited_rgb_run(run,memory,guard,mode):
 value,entry,manifest=audited_png_run(run,memory,guard,'ve')
 raw=(run/'native.log').read_text()
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S)
 if value.get('rgb_buffer')!=mode or len(bodies)!=len(value['requests']):raise RuntimeError('RGB configuration differs')
 for index,(body,request) in enumerate(zip(bodies,value['requests'])):
  reused=int(mode=='resident' and index>0)
  rows=re.findall(r'SD_RGB_BUFFER mode=(request|resident) allocated=(0|1) reused=(0|1) bytes=(\d+)',body)
  expected=dict(mode=mode,allocated=1-reused,reused=reused,bytes=786432)
  if rows!=[(mode,str(1-reused),str(reused),'786432')] or body.count('SD_RGB_BUFFER ')!=1 or request.get('rgb_buffer_dispatch')!=expected:raise RuntimeError('actual RGB allocation/reuse differs')
 entry['rgb_buffer']=mode
 return value,entry,manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results')
 paths=[a.run.resolve(),a.memory.resolve(),a.guard_log.resolve()]
 for path in paths:path.relative_to(ROOT/'build')
 result,entry,manifest=audited_rgb_run(*paths,'resident')
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
 value['publisher_sha256']=sha(Path(__file__));value['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_png_model.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')};value['full_graph_speedup_measured']=False;value['configuration']=dict(rgb_buffer='resident',png_encoder='ve',vae_spatial_tile='8192',im2col_mode='rows_256',generic_threads=8,vae_blas_threads=4,gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');value['scope']='actual-model RGB buffer reuse correctness, PNG O2/main O0, fresh CPU checks; controlled same-binary performance separate'
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Model tests audited:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])
if __name__=='__main__':main()
