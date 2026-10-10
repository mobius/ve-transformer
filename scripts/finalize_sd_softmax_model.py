"""Finalize expanded dispatch validation and bind all guarded audits."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_sd_im2col_result import fan_detail
from record_qwen36_mtp import safe,thermal
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--audit-prefix',required=True);a=p.parse_args()
 proof=a.proof.resolve();prior=a.prior.resolve()
 for path in (proof,prior):path.relative_to(ROOT/'docs/results')
 v=json.loads(proof.read_text());old=json.loads(prior.read_text())
 if not v['all_candidate_tests_completed'] or v['independent_cpu_checks']!=62 or v['cpu_recomputed_checks']!=62 or v['cpu_recomputed_png_checks']!=10 or not old['all_candidate_tests_completed']:raise RuntimeError('complete 62 CPU/10 PNG model validation required')
 if sha(ROOT/'scripts/record_sd_softmax_model.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 for name,digest in v['audit_dependency_sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('audit dependency changed')
 files={};temperatures={};fans={};count=0;oldtests={e['name']:e for e in old['tests']}
 for entry in v['tests']:
  for group in ('sha256','cpu_reference_sha256'):
   for name,digest in entry[group].items():
    if sha(ROOT/name)!=digest:raise RuntimeError('current run/reference changed')
  previous=oldtests[entry['name']]
  for name,digest in previous['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('accepted prior run changed')
  current=json.loads((ROOT/entry['artifacts']/'summary.json').read_text());base=json.loads((ROOT/previous['artifacts']/'summary.json').read_text())
  if current['steps']!=base['steps'] or len(current['requests'])!=len(base['requests']):raise RuntimeError('prior workload differs')
  for r,b in zip(current['requests'],base['requests']):
   if (r['reference_case'],r['trace_sha256'],r['png_sha256'])!=(b['reference_case'],b['trace_sha256'],b['png_sha256']):raise RuntimeError('prior F32/PNG bytes differ')
   count+=1
  log=ROOT/(a.audit_prefix+entry['name']+'-audit.log');log.resolve().relative_to(ROOT/'build')
  if 'Softmax model tests audited:' not in log.read_text():raise RuntimeError('successful CPU audit required')
  paths=guarded_csv(log);temperatures[entry['name']]=thermal(log);fans[entry['name']]=fan_detail(log)
  files.update({str(q.relative_to(ROOT)):sha(q) for q in [log]+paths})
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['model_manifest_sha256']:raise RuntimeError('model changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model input changed')
 kernel=ROOT/'docs/results/20261009T194513Z-sd-turbo-cont-expanded.json';k=json.loads(kernel.read_text())
 objects=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('ve_sd_turbo_cont_transpose.c.o'))
 if len(objects)!=1 or sha(kernel)!=v['expanded_independent_proof_sha256'] or sha(objects[0])!=k['artifact_sha256']['build/sd-cont-expanded-probe/candidate.o'] or sha(objects[0])!=v['actual_cont_compiler']['object_sha256']:raise RuntimeError('actual linked candidate differs from expanded independent test')
 scale=ROOT/'docs/results/20261009T204705Z-sd-turbo-softmax-graph.json';sp=json.loads(scale.read_text());actual=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('ve_sd_turbo_softmax_scale.c.o'))
 if len(actual)!=1 or sha(actual[0])!=sp['artifact_sha256']['build/sd-softmax-graph-probe/scale.o'] or sha(scale)!=v['actual_softmax_scale_compiler']['independent_proof_sha256']:raise RuntimeError('actual tested scale object binding differs')
 snapshot=ROOT/'build/diagnostics/softmax-model-build-20261009T205304Z';snapshot_index=json.loads((snapshot/'archive-sha256.json').read_text())
 for name,digest in snapshot_index.items():
  if sha(snapshot/name)!=digest:raise RuntimeError('build snapshot changed')
 accepted_index=json.loads((ROOT/'build/accepted/softmax-shapes-20261009T201945Z/archive-sha256.json').read_text());math_objects={}
 for source_name,archive_name in (('ve_sd_turbo_softmax.c.o','actual-softmax-exp-object.o'),('ve_sd_turbo_softmax_sum.c.o','actual-softmax-sum-object.o')):
  objects=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob(source_name))
  if len(objects)!=1 or sha(objects[0])!=accepted_index[archive_name]:raise RuntimeError('accepted exponent or sequential sum object changed')
  math_objects[archive_name]=sha(objects[0])
 v['unchanged_accepted_softmax_math_objects']=math_objects;v['build_snapshot_index_sha256']=sha(snapshot/'archive-sha256.json');v['build_snapshot_files']=len(snapshot_index)
 v['previous_model_byte_comparison']=dict(proof=str(prior.relative_to(ROOT)),proof_sha256=sha(prior),requests=count,all_trace_and_png_bytes_identical=True)
 v['actual_model_object_matches_independent_kernel']=True;v['cpu_validation_temperature']=temperatures;v['cpu_validation_fan_detail']=fans;v['cpu_validation_artifact_sha256']=files
 v['model_build_artifact_sha256']={n:sha(ROOT/n) for n in ('build/sd-softmax-model-build.log','build/sd-baseline-ve/manifest.json','build/sd-baseline-ve/bin/sd','scripts/prepare_sd_baseline_overlay.py','scripts/validate_sd_softmax_model.sh','scripts/record_sd_softmax_model.py','docs/results/20261009T194513Z-sd-turbo-cont-expanded.json')}
 v['finalizer_sha256']=sha(Path(__file__));safe(v);proof.write_text(json.dumps(v,indent=2)+'\n');print('Selective Softmax model finalized: 62 CPU/10 PNG, prior bytes identical')
if __name__=='__main__':main()
