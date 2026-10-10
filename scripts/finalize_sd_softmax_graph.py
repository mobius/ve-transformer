"""Bind independent guarded CPU graph audit completion."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');v=json.loads(result.read_text())
 if 'Actual Softmax graph CPU audit passed: '+str(result.relative_to(ROOT)) not in log.read_text() or v['independent_cpu_cases']!=45 or v['timed_graphs']!=240:raise RuntimeError('complete audit required')
 for n,h in v['artifact_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('evidence changed')
 if sha(ROOT/'scripts/record_sd_softmax_graph.py')!=v['publisher_sha256']:raise RuntimeError('auditor changed')
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log);v['artifact_sha256'].update({str(x.relative_to(ROOT)):sha(x) for x in [log]+guarded_csv(log)});v['audit_dependency_sha256']={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_abba.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/benchmark_sd_runtime.py')};v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('Actual Softmax graph independent CPU audit finalized')
if __name__=='__main__':main()
