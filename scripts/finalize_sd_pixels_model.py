"""Pin guarded CPU audits and compare accepted prior-build outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--audit-prefix',required=True);a=p.parse_args()
 proof=a.proof.resolve();prior=a.prior.resolve()
 for path in (proof,prior):path.relative_to(ROOT/'docs/results')
 value=json.loads(proof.read_text());old=json.loads(prior.read_text())
 if not value['all_candidate_tests_completed'] or value['independent_cpu_checks']!=62 or not old['all_candidate_tests_completed']:raise RuntimeError('completed candidate and prior model checks required')
 oldtests={e['name']:e for e in old['tests']};count=0;files={};temperatures={}
 for entry in value['tests']:
  name=entry['name'];previous=oldtests[name]
  for test in (entry,previous):
   for path,digest in test['sha256'].items():
    if sha(ROOT/path)!=digest:raise RuntimeError('model evidence changed')
  current=json.loads((ROOT/entry['artifacts']/'summary.json').read_text());base=json.loads((ROOT/previous['artifacts']/'summary.json').read_text())
  if len(current['requests'])!=len(base['requests']) or current['steps']!=base['steps']:raise RuntimeError('workload differs')
  for r,b in zip(current['requests'],base['requests']):
   if r['reference_case']!=b['reference_case'] or r['trace_sha256']!=b['trace_sha256'] or r['png_sha256']!=b['png_sha256']:raise RuntimeError('prior trace/PNG bytes differ')
   count+=1
  memory=json.loads((ROOT/entry['memory_artifacts']/'summary.json').read_text())
  if memory['final_used_kib']!=131072:raise RuntimeError('node memory did not return to baseline')
  log=ROOT/(a.audit_prefix+name+'-audit.log');log.resolve().relative_to(ROOT/'build');text=log.read_text()
  if 'Model tests audited:' not in text or 'Temperature summary:' not in text or 'Temperature guard stopped command' in text:raise RuntimeError('completed guarded CPU audit required')
  paths=re.findall(r'log=(build/results/[^\s]+\.csv)',text)
  if len(paths)!=2:raise RuntimeError('CPU audit temperature/fan evidence required')
  rows=list(csv.DictReader((ROOT/paths[0]).open()))
  if not rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in rows):raise RuntimeError('unsafe audit temperature')
  temperatures[name]=thermal(log)
  files.update({str(log.relative_to(ROOT)):sha(log)}|{path:sha(ROOT/path) for path in paths})
 value['previous_model_byte_comparison']=dict(proof=str(prior.relative_to(ROOT)),proof_sha256=sha(prior),requests=count,all_trace_and_png_bytes_identical=True)
 value['cpu_validation_temperature']=temperatures;value['cpu_validation_artifact_sha256']=files
 value['finalizer_sha256']=sha(Path(__file__))
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Pixels model finalized: 62 CPU checks,',count,'prior requests byte-identical')
if __name__=='__main__':main()
