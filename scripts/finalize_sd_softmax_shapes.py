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
 if ('Softmax shape/phase profile audited: '+str(result.relative_to(ROOT))) not in log.read_text():raise RuntimeError('completed independent audit required')
 paths=guarded_csv(log)
 if v['status']!='softmax_shape_profile_audited' or len(v['cpu_recomputed_checks'])!=10 or v['cpu_recomputed_png_checks']!=2 or sha(accepted)!=v['accepted_proof_sha256'] or prior['status']!='formal_cont_extended_abba_verified':raise RuntimeError('complete matching current profile audit required')
 if sha(ROOT/'scripts/record_sd_softmax_shapes.py')!=v['publisher_sha256']:raise RuntimeError('auditor changed')
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
 rejection=ROOT/'build/sd-softmax-shapes-node-scope-rejection.log'
 if 'all unique nodes and workers required' not in rejection.read_text():raise RuntimeError('preserved first audit rejection required')
 rejection_paths=guarded_csv(rejection);v['preserved_audit_rejection']=dict(reason='local graph node IDs reused across subgraphs; audit now binds ordered occurrences',temperature=thermal(rejection));v['artifact_sha256'].update({str(q.relative_to(ROOT)):sha(q) for q in [rejection]+rejection_paths})
 v['previous_model_bytes_identical']=True;v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log)
 v['artifact_sha256'].update({str(q.relative_to(ROOT)):sha(q) for q in [log,manifest,accepted]+paths});v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('Softmax shape/phase profile finalized; ten CPU/two PNG and accepted bytes verified')
if __name__=='__main__':main()
