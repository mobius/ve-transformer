"""Bind completed temperature-supervised CPU auditing to refined spatial screening."""
import argparse,hashlib,json,re
from pathlib import Path
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');v=json.loads(proof.read_text())
 if 'Pretranspose microbenchmark audited: '+str(proof.relative_to(ROOT)) not in log.read_text():raise RuntimeError('complete independent CPU audit required')
 if v['status']!='current_baseline_microbenchmark_verified' or v['full_blas_comparisons']!=48 or v['independent_fp64_samples']!=960 or len(v['comparisons'])!=3:raise RuntimeError('complete nine-candidate suite required')
 v['artifact_sha256']={str((ROOT/n).resolve().relative_to(ROOT)):d for n,d in v['artifact_sha256'].items()}
 for n,d in v['artifact_sha256'].items():
  if sha(ROOT/n)!=d:raise RuntimeError('source or raw evidence changed')
 if sha(ROOT/'scripts/record_sd_gemm_pretranspose.py')!=v['publisher_sha256']:raise RuntimeError('CPU publisher changed')
 v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log)
 paths=re.findall(r'Temperature guard:.*log=([^\s]+)',log.read_text())+re.findall(r'Fan observation:.*log=([^\s]+)',log.read_text())
 if len(paths)!=2:raise RuntimeError('one temperature and fan log required')
 for q in [log]+[ROOT/n for n in paths]:
  q.resolve().relative_to(ROOT/'build');v['artifact_sha256'][str(q.relative_to(ROOT))]=sha(q)
 v['decision']='three preparation-inclusive paired gains; model integration only after actual tensor accuracy and image checks; bitwise equality not established'
 v['max_full_blas_difference']=max(r['max_full_difference'] for r in v['records'])
 v['finalizer_sha256']=sha(Path(__file__));safe(v);proof.write_text(json.dumps(v,indent=2)+'\n');print('Guarded refined spatial screening audit finalized')
if __name__=='__main__':main()
