"""Finalize guarded CPU audits and byte comparison for selective small spatial blocks."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_sd_gemm_untiled_model import spatial_compiler_evidence
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--audit-prefix',default='build/sd-gemm-untiled-model-');a=p.parse_args()
 proof=a.proof.resolve();prior=a.prior.resolve()
 for path in (proof,prior):path.relative_to(ROOT/'docs/results')
 v=json.loads(proof.read_text());old=json.loads(prior.read_text())
 if not v['all_candidate_tests_completed'] or v['independent_cpu_checks']!=62 or v['cpu_recomputed_checks']!=62 or v['cpu_recomputed_png_checks']!=10 or not old['all_candidate_tests_completed'] or not old.get('cpu_validation_temperature'):raise RuntimeError('complete guarded model validations required')
 if v['publisher_sha256']!=sha(ROOT/'scripts/record_sd_gemm_untiled_model.py'):raise RuntimeError('auditor changed')
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
  if 'Expanded spatial model tests audited:' not in log.read_text():raise RuntimeError('completed CPU audit required')
  paths=guarded_csv(log);temperatures[entry['name']]=thermal(log);fans[entry['name']]=fan_detail(log);files.update({str(q.relative_to(ROOT)):sha(q) for q in [log]+paths})
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['model_manifest_sha256'] or sha(ROOT/'build/sd-baseline-ve/bin/sd')!=v['binary_sha256'] or sha(ROOT/'tests/check_sd_resident.py')!=v['checker_sha256']:raise RuntimeError('model/checker changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model input changed')
 if spatial_compiler_evidence()!=v['actual_spatial_compiler']:raise RuntimeError('actual spatial build binding changed')
 snapshot=ROOT/'build/diagnostics/gemm-untiled-model-build-20261009T235254Z';index=snapshot/'archive-sha256.json';rows=json.loads(index.read_text())
 for name,digest in rows.items():
  if sha(snapshot/name)!=digest:raise RuntimeError('candidate build snapshot changed')
 v['build_snapshot_index_sha256']=sha(index);v['build_snapshot_files']=len(rows)
 v['previous_model_byte_comparison']=dict(proof=str(prior.relative_to(ROOT)),proof_sha256=sha(prior),requests=count,all_trace_and_png_bytes_identical=True)
 v['cpu_validation_temperature']=temperatures;v['cpu_validation_fan_detail']=fans;v['cpu_validation_artifact_sha256']=files
 v['model_build_artifact_sha256']={n:sha(ROOT/n) for n in ('build/sd-gemm-untiled-model-build.log','build/gemm-untiled-model-precheck.log','build/gemm-untiled-model-object-audit.log','build/sd-baseline-ve/manifest.json','build/sd-baseline-ve/bin/sd','scripts/prepare_sd_gemm_untiled_model.py','scripts/prepare_sd_baseline_overlay.py','scripts/record_sd_baseline_build.py','scripts/validate_sd_gemm_untiled_model.sh','scripts/record_sd_gemm_untiled_model.py','tests/check_sd_resident.py','docs/results/20261009T234838Z-sd-turbo-gemm-untiled.json','build/diagnostics/gemm-untiled-model-before-20261009T235254Z/archive-sha256.json','build/diagnostics/gemm-untiled-model-build-20261009T235254Z/archive-sha256.json')}
 v['finalizer_sha256']=sha(Path(__file__));safe(v);proof.write_text(json.dumps(v,indent=2)+'\n');print('Expanded spatial model finalized: 62 CPU/10 PNG, prior bytes identical')
if __name__=='__main__':main()
