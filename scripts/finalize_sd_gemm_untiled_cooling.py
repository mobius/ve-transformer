"""Bind measured pre-run cooling to the finalized model validation."""
import argparse,json,math
from pathlib import Path
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');v=json.loads(proof.read_text())
 if v.get('status')!='model_validation_verified' or not v.get('cpu_validation_temperature') or not v.get('previous_model_byte_comparison',{}).get('all_trace_and_png_bytes_identical'):raise RuntimeError('finalized model validation required')
 rows={};files={}
 for name in ('double','six','four'):
  path=ROOT/('build/gemm-untiled-model-'+name+'-cooling.json');c=json.loads(path.read_text())
  if any(not math.isfinite(float(c[k])) for k in ('cpu_max_c','ve_max_c','stable_seconds','timeout_seconds','elapsed_seconds','cpu_c','ve_c')):raise RuntimeError('finite cooling evidence required')
  if not c['ready'] or (c['cpu_max_c'],c['ve_max_c'],c['stable_seconds'],c['timeout_seconds'])!=(68,56,10,300) or c['cpu_c']>68 or c['ve_c']>56 or not 10<=c['elapsed_seconds']<=301:raise RuntimeError('cooling policy or measured completion differs')
  guard=ROOT/('build/sd-gemm-untiled-model-'+name+'.log')
  text=guard.read_text()
  if text.count('Measured cool start ready:')!=1 or text.count('Native sequential requests verified:')!=1 or 'Temperature guard stopped command' in text:raise RuntimeError('successful cool-start native run required')
  if text.index('Measured cool start ready:')>text.index('Native sequential requests verified:'):raise RuntimeError('cooling must precede native completion')
  rows[name]=c;files[str(path.relative_to(ROOT))]=sha(path);files[str(guard.relative_to(ROOT))]=sha(guard)
 for name in ('scripts/wait_thermal_ready.py','scripts/run_sd_gemm_untiled_after_cooling.sh','scripts/validate_sd_gemm_untiled_model.sh','scripts/finalize_sd_gemm_untiled_cooling.py'):files[name]=sha(ROOT/name)
 v['pre_run_cooling']=rows;v['cooling_artifact_sha256']=files;safe(v);proof.write_text(json.dumps(v,indent=2)+'\n');print('Expanded model cooling evidence finalized')
if __name__=='__main__':main()
