"""Bind CPU completion, dependency versions and preserved environment rejection."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');v=json.loads(result.read_text())
 if ('Strict sum CPU audit passed: '+str(result.relative_to(ROOT))) not in log.read_text() or v['independent_cpu_checks']!=108 or v['timed_batches']!=120:raise RuntimeError('complete independent CPU audit required')
 paths=guarded_csv(log)
 for n,h in v['artifact_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('evidence changed')
 if sha(ROOT/'scripts/record_sd_softmax_sum.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 reject=ROOT/'build/sd-softmax-sum-audit-environment-rejection.log'
 if "No module named 'PIL'" not in reject.read_text():raise RuntimeError('preserved environment failure required')
 extra=[log]+paths+[reject]+guarded_csv(reject)
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log);v['preserved_rejection']='CPU audit dependency absent in base venv; rerun using existing project image venv, no hardware rerun'
 v['artifact_sha256'].update({str(x.relative_to(ROOT)):sha(x) for x in extra});v['audit_dependency_sha256']={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_abba.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/benchmark_sd_runtime.py')};v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('Strict sum independent CPU audit finalized')
if __name__=='__main__':main()
