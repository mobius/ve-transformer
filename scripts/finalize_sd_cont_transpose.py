"""Bind successful CPU-audit thermal evidence and preserved preparation/audit failures."""
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
 if ('CONT transpose microbenchmark audited: '+str(result.relative_to(ROOT))) not in log.read_text():raise RuntimeError('successful completed CPU audit required')
 paths=guarded_csv(log);v=json.loads(result.read_text())
 if v['status']!='cont_transpose_microbenchmark_verified' or v['cpu_bit_checks']!=6 or v['timed_graphs']!=72 or v['checks']!={'ownership':180,'concurrent':48,'graph':24,'invalid':14}:raise RuntimeError('complete transpose checks required')
 for group in ('artifact_sha256','audit_dependency_sha256'):
  for name,digest in v[group].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('independent test evidence changed')
 if sha(ROOT/'scripts/record_sd_cont_transpose.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['baseline']['manifest_sha256']:raise RuntimeError('model changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model source changed')
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log);files=[log]+paths
 v['preserved_rejections']=[]
 for name,reason in (('build/sd-cont-transpose-fixture-rejection.log','pre-execution singleton-dimension stride check was too strict; fixed before compilation/VE execution'),('build/sd-cont-transpose-audit-thread-log-rejection.log','audit incorrectly expected a diagnostic thread marker in the actual ggml single-thread fast path; corrected without rerunning hardware')):
  path=ROOT/name;files.extend([path]+guarded_csv(path));v['preserved_rejections'].append(dict(log=name,reason=reason,temperature=thermal(path)))
 v['artifact_sha256'].update({str(path.relative_to(ROOT)):sha(path) for path in files});v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('CONT transpose guarded CPU audit finalized; model unchanged')
if __name__=='__main__':main()
