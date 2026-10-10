"""Bind completed guarded CPU audit temperature and unchanged diagnostic evidence."""
import argparse
import json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build')
 text=log.read_text()
 if ('CONT shape profile audited: '+str(result.relative_to(ROOT))) not in text:raise RuntimeError('complete CPU audit required')
 paths=guarded_csv(log);v=json.loads(result.read_text())
 if v['status']!='cont_shape_profile_verified' or v['independent_cpu_checks']!=10 or v['cpu_recomputed_png_checks']!=2 or not v['previous_model_bytes_identical']:raise RuntimeError('complete diagnostic audit required')
 for group in ('artifact_sha256','audit_dependency_sha256','cpu_reference_sha256'):
  for name,digest in v[group].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('audit evidence changed')
 if sha(ROOT/'scripts/record_sd_cont_shapes.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['manifest_sha256']:raise RuntimeError('model changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model input changed')
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log)
 v['artifact_sha256'].update({str(path.relative_to(ROOT)):sha(path) for path in [log]+paths})
 v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('CONT guarded CPU audit finalized')
if __name__=='__main__':main()
