"""Audit actual-model selective spatial tiling and independent CPU references."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch
ROOT=Path(__file__).resolve().parents[1]
EXPECTED=Counter({('vae',256,262144,2304,8192,32):1,('vae',512,65536,4608,8192,8):1,('vae',256,65536,2304,8192,8):5,('vae',256,65536,4608,8192,8):1,('vae',128,262144,2304,8192,32):1})
def tile_dispatch(folder,value,mode):
 raw=(folder/'native.log').read_text()
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S)
 if len(bodies)!=len(value['requests']) or value.get('vae_spatial_tile')!=mode:raise RuntimeError('tiling mode/request count differs')
 for body,request in zip(bodies,value['requests']):
  rows=re.findall(r'SD_NLC_SPATIAL_TILE stage=(vae) m=(\d+) n=(\d+) k=(\d+) tile=(\d+) calls=(\d+)',body)
  observed=Counter((r[0],)+tuple(map(int,r[1:])) for r in rows)
  parsed=[dict(stage=r[0],m=int(r[1]),n=int(r[2]),k=int(r[3]),tile=int(r[4]),calls=int(r[5])) for r in rows]
  if observed!=(EXPECTED if mode=='8192' else Counter()) or len(rows)!=body.count('SD_NLC_SPATIAL_TILE ') or parsed!=request.get('vae_spatial_dispatch'):raise RuntimeError('actual tiling shape/call dispatch differs')
def audited_run(run,memory,guard,mode):
 value=json.loads((run/'summary.json').read_text());manifest_path=ROOT/'build/sd-baseline-ve/manifest.json';manifest=json.loads(manifest_path.read_text())
 for name,digest in manifest['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model manifest changed')
 if (value['gelu_mode']!='ve' or value['binary_scalar_mode']!='ve' or value['vae_threads']!=8 or value['vae_blas_threads']!=4 or value['nlc_threads']!='unified' or value['tokenizer_mode']!='resident' or value['mode']!='resident' or value.get('operator_profile_enabled') or value.get('binary_shape_profile_enabled') or value.get('im2col_mode')!='rows_256'):raise RuntimeError('fixed optimized configuration differs')
 checks=check_requests(run,value,manifest['sha256']['build/sd-baseline-ve/bin/sd']);rows_dispatch(run,value,'rows_256');tile_dispatch(run,value,mode)
 if 'SD_NLC_GEMM ' in (run/'native.log').read_text():raise RuntimeError('unexpected per-call timing probe')
 sampled=json.loads((memory/'summary.json').read_text());rows=list(csv.DictReader((memory/'memory.csv').open()))
 if not sampled['completed'] or sampled['returncode'] or sampled['final_used_kib']!=131072 or not rows or int(rows[-1]['used_kib'])!=131072 or max(int(r['used_kib']) for r in rows)!=sampled['sampled_highest_used_kib']:raise RuntimeError('memory completion/peak/recovery differs')
 paths=re.findall(r'log=(build/results/[^\s]+\.csv)',guard.read_text())
 if len(paths)!=2:raise RuntimeError('thermal and fan CSVs required')
 samples=list(csv.DictReader((ROOT/paths[0]).open()))
 if not samples or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in samples):raise RuntimeError('thermal stop or invalid sample')
 files=[run/'summary.json',run/'native.log',run/'check_sd_resident.py',memory/'summary.json',memory/'memory.csv',guard]+[ROOT/p for p in paths]
 entry=dict(artifacts=str(run.relative_to(ROOT)),memory_artifacts=str(memory.relative_to(ROOT)),sha256={str(p.relative_to(ROOT)):sha(p) for p in files},request_seconds=[r['request_seconds'] for r in value['requests']],cpu_checks=checks,temperature=thermal(guard),fan_detail=fan_detail(guard),sampled_node_peak_gib=sampled['sampled_highest_used_kib']/1048576)
 return value,entry,sha(manifest_path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results')
 paths=[a.run.resolve(),a.memory.resolve(),a.guard_log.resolve()]
 for path in paths:path.relative_to(ROOT/'build')
 result,entry,manifest=audited_run(*paths,'8192')
 steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[a.name]
 if result['steps']!=steps or [r['reference_case'] for r in result['requests']]!=cases:raise RuntimeError('wrong workload')
 value=json.loads(proof.read_text()) if proof.exists() else dict(status='model_validation_in_progress',tests=[],model_manifest_sha256=manifest,binary_sha256=result['binary_sha256'],checker_sha256=result['checker_sha256'])
 if value['model_manifest_sha256']!=manifest or value['binary_sha256']!=result['binary_sha256'] or value['checker_sha256']!=result['checker_sha256']:raise RuntimeError('candidate build/checker changed')
 existing=[e for e in value['tests'] if e['name']!=a.name]
 for e in existing:
  for name,digest in e['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('earlier test evidence changed')
 entry['name']=a.name;value['tests']=sorted(existing+[entry],key=lambda e:('double','six','four').index(e['name']))
 value['completed_tests']=[e['name'] for e in value['tests']];value['independent_cpu_checks']=sum(e['cpu_checks'] for e in value['tests']);value['all_candidate_tests_completed']=value['completed_tests']==['double','six','four'];value['status']='model_validation_verified' if value['all_candidate_tests_completed'] else 'model_validation_in_progress'
 value['publisher_sha256']=sha(Path(__file__));value['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')};value['full_graph_speedup_measured']=False;value['configuration']=dict(vae_spatial_tile='8192',im2col_mode='rows_256',generic_threads=8,vae_blas_threads=4,gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');value['scope']='new actual-model candidate correctness, no old-build CPU checks; controlled same-binary performance separate'
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Model tests audited:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])
if __name__=='__main__':main()
