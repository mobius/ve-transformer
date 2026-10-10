"""Bind guarded CPU completion and byte comparison for diagnostic evidence."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--accepted',type=Path,required=True);a=p.parse_args()
 result=a.result.resolve();accepted=a.accepted.resolve();log=a.log.resolve();native=a.native.resolve()
 for q in (result,accepted):q.relative_to(ROOT/'docs/results')
 for q in (log,native):q.relative_to(ROOT/'build')
 v=json.loads(result.read_text());prior=json.loads(accepted.read_text())
 if ('GroupNorm shape profile audited: '+str(result.relative_to(ROOT))) not in log.read_text():raise RuntimeError('completed independent audit required')
 paths=guarded_csv(log)
 if v['status'] not in ('group_norm_shape_profile_audited','group_norm_shape_profile_verified') or len(v['cpu_recomputed_checks'])!=10 or v['cpu_recomputed_png_checks']!=2 or sha(accepted)!=v['accepted_proof_sha256'] or prior['status']!='formal_softmax_scale_abba_verified':raise RuntimeError('complete matching current profile audit required')
 if sha(ROOT/'scripts/record_sd_group_norm_shapes.py')!=v['publisher_sha256']:raise RuntimeError('auditor changed')
 for group in ('artifact_sha256','audit_helper_sha256','cpu_reference_sha256'):
  for name,digest in v[group].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('diagnostic evidence changed')
 value=json.loads((native/'summary.json').read_text());base=json.loads((ROOT/prior['runs'][0]['artifacts']/'summary.json').read_text())
 if len(value['requests'])!=2 or value['steps']!=1:raise RuntimeError('two one-step diagnostic requests required')
 for r,b in zip(value['requests'],base['requests']):
  if (r['reference_case'],r['trace_sha256'],r['png_sha256'])!=(b['reference_case'],b['trace_sha256'],b['png_sha256']):raise RuntimeError('profile output differs from accepted formal output')
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['manifest_sha256']:raise RuntimeError('model changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model input changed')
 rejection=ROOT/'build/sd-group-norm-shapes-argument-rejection.log'
 if 'unrecognized arguments: --operator-profile' not in rejection.read_text():raise RuntimeError('preserved argument rejection required')
 v['preserved_argument_rejection']=dict(reason='incorrect diagnostic option rejected before VE inference; corrected op-profile, same build',temperature=thermal(rejection));v['artifact_sha256'].update({str(q.relative_to(ROOT)):sha(q) for q in [rejection]+guarded_csv(rejection)})
 index=json.loads((ROOT/'build/accepted/softmax-scale-abba-20261009T210240Z/archive-sha256.json').read_text());objects={}
 for source_name,archive_name in (('ve_sd_turbo_softmax.c.o','actual-softmax-exp-object.o'),('ve_sd_turbo_softmax_sum.c.o','actual-softmax-sum-object.o'),('ve_sd_turbo_softmax_scale.c.o','actual-softmax-scale-object.o')):
  math_paths=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob(source_name))
  if len(math_paths)!=1 or sha(math_paths[0])!=index[archive_name]:raise RuntimeError('accepted Softmax math object changed')
  objects[archive_name]=sha(math_paths[0])
 v['unchanged_accepted_math_objects']=objects;v['model_build_log_sha256']=sha(ROOT/'build/sd-group-norm-shapes-build.log')
 v['status']='group_norm_shape_profile_verified';v['previous_model_bytes_identical']=True;v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log)
 v['artifact_sha256'].update({str(q.relative_to(ROOT)):sha(q) for q in [log,manifest,accepted]+paths});v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('GroupNorm shape profile finalized; ten CPU/two PNG and accepted bytes verified')
if __name__=='__main__':main()
