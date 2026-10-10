"""Bind completed guarded CPU audit and frozen spatial probe evidence."""
import argparse,json
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args()
 result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');v=json.loads(result.read_text())
 if ('Small spatial tiling microbenchmark audited: '+str(result.relative_to(ROOT))) not in log.read_text() or v['status']!='current_baseline_microbenchmark_verified' or v['full_blas_comparisons']!=288 or v['independent_fp64_samples']!=5760 or v['baseline_tiles']!=[0,8192,8192,8192,8192,8192]:raise RuntimeError('complete current-baseline CPU audit required')
 paths=guarded_csv(log)
 for n,h in v['artifact_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('evidence changed')
 if sha(ROOT/'scripts/record_sd_gemm_spatial_small.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log)
 v['artifact_sha256'].update({str(x.relative_to(ROOT)):sha(x) for x in [log]+paths})
 v['audit_dependency_sha256']={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_abba.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/benchmark_sd_runtime.py','scripts/benchmark_nlc_gemm.py')}
 v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('Small spatial tiling guarded CPU audit finalized')
if __name__=='__main__':main()
