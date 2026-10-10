"""Bind guarded CPU completion and preserved strict IEEE failures."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');v=json.loads(result.read_text())
 if 'Softmax scale CPU audit passed: '+str(result.relative_to(ROOT)) not in log.read_text() or v['independent_cpu_output_checks']!=288 or v['timed_batches']!=240 or v['ieee_strict_reference_pass'] is not False:raise RuntimeError('complete FTZ audit and explicit strict rejection required')
 for n,h in v['artifact_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('evidence changed')
 if sha(ROOT/'scripts/record_sd_softmax_scale.py')!=v['publisher_sha256']:raise RuntimeError('auditor changed')
 extra=[log]+guarded_csv(log)
 for name in ('build/sd-softmax-scale-audit-rounding-rejection.log','build/sd-softmax-scale-audit-diagnostic.log'):
  rejection=ROOT/name
  if 'RuntimeError: independent float32 output differs' not in rejection.read_text():raise RuntimeError('strict IEEE rejection missing')
  extra += [rejection]+guarded_csv(rejection)
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log);v['artifact_sha256'].update({str(x.relative_to(ROOT)):sha(x) for x in extra});v['audit_dependency_sha256']={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_abba.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/benchmark_sd_runtime.py')};v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('Softmax scale FTZ audit finalized; strict IEEE reference remains rejected')
if __name__=='__main__':main()
