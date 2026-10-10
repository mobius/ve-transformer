"""Attach completed guarded CPU validation to the independently audited CONT ABBA."""
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
 if ('CONT formal ABBA audited: '+str(result.relative_to(ROOT))) not in log.read_text():raise RuntimeError('successful completed CPU audit required')
 paths=guarded_csv(log);v=json.loads(result.read_text())
 if v['status']!='formal_cont_abba_verified' or v['independent_cpu_checks']!=142 or v['cpu_recomputed_png_checks']!=26 or not v['same_binary'] or not v['all_trace_and_png_bytes_identical']:raise RuntimeError('complete same-binary formal audit required')
 for group in ('artifact_sha256','audit_dependency_sha256'):
  for name,digest in v[group].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('formal evidence changed')
 for entry in v['runs']+v['prior_runs']:
  for group in ('sha256','cpu_reference_sha256'):
   for name,digest in entry[group].items():
    if sha(ROOT/name)!=digest:raise RuntimeError('native/reference evidence changed')
 if sha(ROOT/'scripts/record_sd_cont_abba.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=v['model_manifest_sha256']:raise RuntimeError('model changed')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model input changed')
 failed=ROOT/'build/diagnostics/cont-abba-thermal-stop-20261009T191953Z/runtime.log'
 failed_summary=ROOT/'build/results/20261009T192011Z-sd-cont-abba/summary.json'
 failure=json.loads(failed_summary.read_text())
 if failure['completed'] or len(failure['runs'])!=3 or 'temperature reached stop threshold' not in failed.read_text():raise RuntimeError('preserved incomplete thermal-stop evidence required')
 failure_files=[failed,failed_summary,failed.parent/'benchmark_sd_cont_abba.py',ROOT/'build/results/20261009T192011Z-sd-cont-abba/benchmark_sd_cont_abba.py',ROOT/'build/results/20261009T192011Z-temperature-ees15ieu.csv',ROOT/'build/results/20261009T192011Z-temperature-ees15ieu-fans.csv']
 v['preserved_failed_attempt']=dict(completed=False,completed_arms=3,reason='CPU 85C temperature stop; incomplete ABBA excluded from formal means',temperature=thermal(failed),fan_detail=fan_detail(failed),artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in failure_files})
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log);v['artifact_sha256'].update({str(path.relative_to(ROOT)):sha(path) for path in [log]+paths});v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('CONT formal guarded CPU audit finalized')
if __name__=='__main__':main()
