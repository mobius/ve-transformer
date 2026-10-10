"""Pin guarded CPU audits and compare accepted prior-build outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--audit-prefix',required=True);a=p.parse_args()
 proof=a.proof.resolve();prior=a.prior.resolve()
 for path in (proof,prior):path.relative_to(ROOT/'docs/results')
 value=json.loads(proof.read_text());old=json.loads(prior.read_text())
 if value.get('cpu_recomputed_checks')!=62 or value.get('cpu_recomputed_png_checks')!=10:raise RuntimeError('full independent recomputed CPU/PNG checks required')
 if not value['all_candidate_tests_completed'] or value['independent_cpu_checks']!=62 or not old['all_candidate_tests_completed']:raise RuntimeError('completed candidate and prior model checks required')
 oldtests={e['name']:e for e in old['tests']};count=0;files={};temperatures={}
 for entry in value['tests']:
  name=entry['name'];previous=oldtests[name]
  for test in (entry,previous):
   for path,digest in test['sha256'].items():
    if sha(ROOT/path)!=digest:raise RuntimeError('model evidence changed')
  current=json.loads((ROOT/entry['artifacts']/'summary.json').read_text());base=json.loads((ROOT/previous['artifacts']/'summary.json').read_text())
  if len(current['requests'])!=len(base['requests']) or current['steps']!=base['steps']:raise RuntimeError('workload differs')
  for r,b in zip(current['requests'],base['requests']):
   if r['reference_case']!=b['reference_case'] or r['trace_sha256']!=b['trace_sha256'] or r['png_sha256']!=b['png_sha256']:raise RuntimeError('prior trace/PNG bytes differ')
   count+=1
  memory=json.loads((ROOT/entry['memory_artifacts']/'summary.json').read_text())
  if memory['final_used_kib']!=131072:raise RuntimeError('node memory did not return to baseline')
  log=ROOT/(a.audit_prefix+name+'-audit.log');log.resolve().relative_to(ROOT/'build');text=log.read_text()
  if 'CONT model tests audited:' not in text or 'Temperature summary:' not in text or 'Temperature guard stopped command' in text:raise RuntimeError('completed guarded CPU audit required')
  paths=re.findall(r'log=(build/results/[^\s]+\.csv)',text)
  if len(paths)!=2:raise RuntimeError('CPU audit temperature/fan evidence required')
  rows=list(csv.DictReader((ROOT/paths[0]).open()))
  if not rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in rows):raise RuntimeError('unsafe audit temperature')
  temperatures[name]=thermal(log)
  files.update({str(log.relative_to(ROOT)):sha(log)}|{path:sha(ROOT/path) for path in paths})
 value['previous_model_byte_comparison']=dict(proof=str(prior.relative_to(ROOT)),proof_sha256=sha(prior),requests=count,all_trace_and_png_bytes_identical=True)
 value['cpu_validation_temperature']=temperatures;value['cpu_validation_artifact_sha256']=files
 kernel_path=ROOT/'docs/results/20261009T184817Z-sd-turbo-cont-transpose.json';kernel=json.loads(kernel_path.read_text())
 if sha(kernel_path)!=value['actual_cont_compiler']['independent_proof_sha256'] or kernel['artifact_sha256']['build/sd-cont-transpose-probe/candidate.o']!=value['actual_cont_compiler']['object_sha256']:raise RuntimeError('actual model object differs from independently tested object')
 obj=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('ve_sd_turbo_cont_transpose.c.o'))
 if len(obj)!=1 or sha(obj[0])!=value['actual_cont_compiler']['object_sha256']:raise RuntimeError('actual object changed')
 if value['publisher_sha256']!=sha(ROOT/'scripts/record_sd_cont_model.py'):raise RuntimeError('CPU auditor changed')
 for name,digest in value['audit_dependency_sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('CPU audit dependency changed')
 for entry in value['tests']:
  for name,digest in entry['cpu_reference_sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('CPU reference changed')
 value['actual_model_object_matches_independent_kernel']=True
 value['model_build_artifact_sha256']={name:sha(ROOT/name) for name in ('build/sd-cont-model-build.log','build/sd-cont-model-object-build.log','build/sd-baseline-ve/manifest.json','build/sd-baseline-ve/bin/sd','cmake/nec-sd-baseline-overrides.cmake','scripts/validate_sd_cont_model.sh','scripts/record_sd_cont_model.py','scripts/finalize_sd_cont_model.py','docs/results/20261009T184817Z-sd-turbo-cont-transpose.json')}
 failure_folder=ROOT/'build/diagnostics/cont-model-missed-dispatch-20261009T185903Z'
 failure_index=failure_folder/'archive-sha256.json';failure_files=json.loads(failure_index.read_text())
 for name,digest in failure_files.items():
  if sha(failure_folder/name)!=digest:raise RuntimeError('preserved zero-dispatch evidence changed')
 failed_run=failure_folder/'build/results/20261009T190237Z-sd-resident'
 failed=json.loads((failed_run/'summary.json').read_text());markers=re.findall(r'SD_CONT_DISPATCH stage=(clip|unet|vae) optimized=(\d+) fallback=(\d+) enabled=(0|1)',(failed_run/'native.log').read_text())
 if markers!=[('clip','0','0','1'),('unet','0','0','1'),('vae','0','0','1')]*2:raise RuntimeError('preserved failure differs from zero dispatch')
 failed_log=failure_folder/'build/sd-cont-model-double.log';failed_audit=failure_folder/'build/sd-cont-model-double-audit.log'
 if 'actual CONT candidate/fallback count differs' not in failed_audit.read_text():raise RuntimeError('original strict rejection required')
 value['preserved_failed_dispatch']=dict(artifacts=str(failure_folder.relative_to(ROOT)),archive_sha256=sha(failure_index),request_seconds=[r['request_seconds'] for r in failed['requests']],candidate_not_executed=True,reason='same-type CONT used dup_bytes before the wrongly hooked dup_f32 branch; strict dispatch audit rejected; fixed in public dup and reran all model tests',temperature=thermal(failed_log),cpu_audit_temperature=thermal(failed_audit))
 value['failure_artifact_sha256']={str(path.relative_to(ROOT)):sha(path) for path in (failure_index,failed_run/'summary.json',failed_run/'native.log',failed_log,failed_audit)}
 value['finalizer_sha256']=sha(Path(__file__))
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('CONT model finalized: 62 CPU checks,',count,'prior requests byte-identical')
if __name__=='__main__':main()
