"""Finalize guarded CPU audits and byte comparison for selective small spatial blocks."""
import argparse,json,re
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_sd_current_vae_build import evidence as current_build_evidence,retained_compilers
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--audit-prefix',default='build/sd-current-vae-threads-');a=p.parse_args()
 proof=a.proof.resolve();prior=a.prior.resolve()
 for path in (proof,prior):path.relative_to(ROOT/'docs/results')
 v=json.loads(proof.read_text());old=json.loads(prior.read_text())
 if not v['all_candidate_tests_completed'] or v['independent_cpu_checks']!=62 or v['cpu_recomputed_checks']!=62 or v['cpu_recomputed_png_checks']!=10 or not old['all_candidate_tests_completed'] or not old.get('cpu_validation_temperature'):raise RuntimeError('complete guarded model validations required')
 if v['publisher_sha256']!=sha(ROOT/'scripts/record_sd_current_vae_threads.py'):raise RuntimeError('auditor changed')
 for name,digest in v['audit_dependency_sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('audit dependency changed')
 files={};temperatures={};fans={};oldtests={e['name']:e for e in old['tests']};count=0
 for entry in v['tests']:
  for group in ('sha256','cpu_reference_sha256'):
   for name,digest in entry[group].items():
    if sha(ROOT/name)!=digest:raise RuntimeError('current native or reference evidence changed')
  previous=oldtests[entry['name']]
  for name,digest in previous['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('accepted previous request evidence changed')
  current=json.loads((ROOT/entry['artifacts']/'summary.json').read_text());base=json.loads((ROOT/previous['artifacts']/'summary.json').read_text())
  if current['steps']!=base['steps'] or len(current['requests'])!=len(base['requests']):raise RuntimeError('previous workload differs')
  for r,b in zip(current['requests'],base['requests']):
   if (r['reference_case'],r['trace_sha256'],r['png_sha256'])!=(b['reference_case'],b['trace_sha256'],b['png_sha256']):raise RuntimeError('previous F32/PNG bytes differ')
   count+=1
  log=ROOT/(a.audit_prefix+entry['name']+'-audit.log');log.resolve().relative_to(ROOT/'build')
  if 'Current VAE thread model tests audited:' not in log.read_text():raise RuntimeError('completed CPU audit required')
  paths=guarded_csv(log);temperatures[entry['name']]=thermal(log);fans[entry['name']]=fan_detail(log);files.update({str(q.relative_to(ROOT)):sha(q) for q in [log]+paths})
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['model_manifest_sha256'] or sha(ROOT/'build/sd-baseline-ve/bin/sd')!=v['binary_sha256'] or sha(ROOT/'tests/check_sd_resident.py')!=v['checker_sha256']:raise RuntimeError('model/checker changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model input changed')
 if current_build_evidence()!=v['actual_thread_dispatch_compiler'] or any(v[k]!=x for k,x in retained_compilers().items()):raise RuntimeError('actual dispatch or retained mathematical build binding changed')
 v['previous_model_byte_comparison']=dict(proof=str(prior.relative_to(ROOT)),proof_sha256=sha(prior),requests=count,all_trace_and_png_bytes_identical=True)
 v['cpu_validation_temperature']=temperatures;v['cpu_validation_fan_detail']=fans;v['cpu_validation_artifact_sha256']=files
 accepted=ROOT/'docs/results/20261010T000834Z-sd-turbo-gemm-untiled-abba.json';base=json.loads(accepted.read_text())
 if sha(accepted)!=v['accepted_current_proof_sha256'] or base['checker_sha256']!=v['checker_sha256'] or v['model_code_unchanged'] or not v['mathematical_kernel_objects_unchanged']:raise RuntimeError('accepted prior model and exact dispatch-only delta required')
 v['runtime_artifact_sha256']={n:sha(ROOT/n) for n in ('build/current-vae-threads-precheck.log','build/sd-baseline-ve/manifest.json','build/sd-baseline-ve/bin/sd','scripts/validate_sd_current_vae_threads.sh','scripts/run_sd_current_vae_after_cooling.sh','scripts/record_sd_current_vae_threads.py','tests/check_sd_resident.py','docs/results/20261010T000834Z-sd-turbo-gemm-untiled-abba.json')}
 failed=ROOT/'build/diagnostics/current-vae-first-stop-20261010T002648Z'
 index=json.loads((failed/'archive-sha256.json').read_text())
 for name,digest in index.items():
  if sha(failed/name)!=digest:raise RuntimeError('failed prototype evidence changed')
 first=json.loads((failed/'build/results/20261010T002934Z-sd-resident/summary.json').read_text())
 if first['completed'] or first['returncode']!=0 or len(first['requests'])!=2:raise RuntimeError('initial coverage failure provenance differs')
 v['failed_initial_prototype']=dict(reason='old eight-thread dispatch gate gave VAE rows=0/channels=40; checker rejected coverage; independent CPU audit not run',native_returncode=0,checker_returncode=1,request_seconds=[float(t) for t in re.findall(r'SD_REQUEST_END index=\d+ seconds=([0-9.]+)',(failed/'build/results/20261010T002934Z-sd-resident/native.log').read_text())],frozen_archive=str(failed.relative_to(ROOT)),index_sha256=sha(failed/'archive-sha256.json'),temperature=thermal(failed/'build/sd-current-vae-threads-double.log'),fan_detail=fan_detail(failed/'build/sd-current-vae-threads-double.log'))
 v['build_temperature']=thermal(ROOT/'build/sd-current-vae-gate-build.log');v['build_fan_detail']=fan_detail(ROOT/'build/sd-current-vae-gate-build.log')
 v['build_and_file_audit_logs_sha256']={n:sha(ROOT/n) for n in ('build/sd-current-vae-gate-build.log','build/sd-current-vae-build-audit.log','build/sd-current-vae-build-audit-retry.log','build/sd-current-vae-build-final-audit.log')}
 v['finalizer_sha256']=sha(Path(__file__));safe(v);proof.write_text(json.dumps(v,indent=2)+'\n');print('Current VAE thread model finalized: 62 CPU/10 PNG, prior bytes identical')
if __name__=='__main__':main()
