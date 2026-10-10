"""Audit actual-ggml CONT microbenchmark, ownership/concurrency and full CPU bit copies."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import numpy as np
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();folder=a.native.resolve();folder.relative_to(ROOT/'build');guard=a.guard_log.resolve();guard.relative_to(ROOT/'build');target=a.output.resolve();target.relative_to(ROOT/'docs/results')
 log=guard.read_text()
 if 'CONT_TRANSPOSE_PASS real_shapes=3 direct_shapes=12 invalid=14 timing_rows=72' not in log or ('CONT transpose artifacts: '+str(folder.relative_to(ROOT))) not in log:raise RuntimeError('complete VE graph/direct test required')
 thermal_paths=guarded_csv(guard);binding_path=ROOT/'build/sd-cont-transpose-probe/baseline.json';binding=json.loads(binding_path.read_text());proof=ROOT/'docs/results/20261009T183852Z-sd-turbo-cont-shapes.json'
 if sha(proof)!=binding['proof_sha256'] or sha(ROOT/'build/sd-baseline-ve/manifest.json')!=binding['manifest_sha256'] or sha(ROOT/'build/sd-baseline-ve/bin/sd')!=binding['binary_sha256']:raise RuntimeError('actual baseline changed')
 for name,digest in binding['libraries_sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('actual graph archive changed')
 objects=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('sd-ggml-cpu.c.o'))
 if len(objects)!=1 or sha(objects[0])!=binding['actual_cpu_object_sha256']:raise RuntimeError('actual model CPU object changed')
 shapes=binding['shapes'];expected_shapes=json.loads(proof.read_text())['hot_shape_groups'][:3]
 for row,observed in zip(shapes,expected_shapes):
  if row['shape']!=observed['a'] or row['strides']!=observed['an'] or row['nodes']!=observed['nodes'] or row['diagnostic_seconds']!=observed['seconds']:raise RuntimeError('actual top layouts differ')
 fixture_path=ROOT/'build/sd-cont-transpose-probe/shapes.tsv'
 expected_tsv=''.join('\t'.join(map(str,[i]+r['shape']+r['strides']))+'\n' for i,r in enumerate(shapes))
 if fixture_path.read_text()!=expected_tsv:raise RuntimeError('driver geometry differs')
 dimensions=[(r['m'],r['n']) for r in shapes]+[(17,n) for n in (1,2,7,255,256,257,4095,4096)]+[(1,1)]
 checks=list(csv.reader((folder/'checks.tsv').open(),delimiter='\t'));expected=[]
 for ident,(m,n) in enumerate(dimensions):
  for nth in (1,2,4,8):
   dr=(n+nth-1)//nth
   for ith in range(nth):expected.append(['ownership',str(ident),str(nth),str(ith),str(dr*ith),str(min(dr*(ith+1),n)),'PASS'])
   expected.append(['concurrent',str(ident),str(nth),'PASS'])
  if ident<3:
   for nth in (1,2,4,8):
    for mode in ('baseline','candidate'):expected.append(['graph',str(ident),str(nth),mode,'PASS'])
 expected += [['invalid',str(i),'PASS'] for i in range(14)]
 if checks!=expected:raise RuntimeError('complete ordered ownership/concurrent/actual-graph/invalid checks required')
 timings=list(csv.reader((folder/'timings.tsv').open(),delimiter='\t'));order=[[str(i),str(rep),str(arm),'candidate' if arm in (1,2) else 'baseline','8'] for i in range(3) for rep in range(6) for arm in range(4)]
 if len(timings)!=72 or [r[:-1] for r in timings]!=order or any(not math.isfinite(float(r[-1])) or float(r[-1])<=0 for r in timings):raise RuntimeError('bounded ordered actual graph ABBA timings required')
 actual=list(map(int,re.findall(r'SD_GGML_THREADS stage=unknown actual=(\d+) maximum=\d+',log)))
 expected_actual=[]
 for i in range(3):
  for nth in (1,2,4,8):
   if nth>1:expected_actual.extend([nth,nth])
   if nth==8:expected_actual.extend([8]*24)
 if actual!=expected_actual:raise RuntimeError('actual baseline/candidate ggml teams differ')
 results=[];files=[folder/'checks.tsv',folder/'timings.tsv',guard,binding_path,fixture_path,proof,ROOT/'build/sd-baseline-ve/manifest.json',ROOT/'build/sd-baseline-ve/bin/sd']+thermal_paths
 edges=np.asarray([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00001,0xffc12345,0x7fffffff,0xffffffff],dtype=np.uint32)
 for i,row in enumerate(shapes):
  input_path=folder/('shape%d-input.f32'%i);source=np.fromfile(input_path,dtype=np.uint32)
  if source.size!=row['m']*row['n']:raise RuntimeError('complete input bits required')
  for offset,value in enumerate(edges):
   if not np.all(source[offset::31]==value):raise RuntimeError('edge bit-pattern coverage differs')
  expected=source.reshape(row['m'],row['n']).T.copy().reshape(-1)
  files.append(input_path)
  for mode in ('baseline','candidate'):
   path=folder/('shape%d-%s.f32'%(i,mode));output=np.fromfile(path,dtype=np.uint32)
   if not np.array_equal(output,expected):raise RuntimeError('independently recomputed full transpose bits differ')
   files.append(path)
  samples={mode:[float(r[-1]) for r in timings if int(r[0])==i and r[3]==mode] for mode in ('baseline','candidate')};means={mode:sum(t)/len(t) for mode,t in samples.items()}
  pairs=[]
  for left,right in ((0,1),(3,2)):
   b=sum(float(r[-1]) for r in timings if int(r[0])==i and int(r[2])==left)/6;c=sum(float(r[-1]) for r in timings if int(r[0])==i and int(r[2])==right)/6;pairs.append(dict(baseline_arm=left,candidate_arm=right,baseline_seconds=b,candidate_seconds=c,latency_reduction_percent=(1-c/b)*100))
  results.append({**row,'baseline_seconds':means['baseline'],'candidate_seconds':means['candidate'],'latency_reduction_percent':(1-means['candidate']/means['baseline'])*100,'samples_seconds':samples,'paired_arm_means':pairs,'all_cpu_uint32_bits_identical':True,'cpu_bit_checks':2})
 files.extend(ROOT/n for n in binding['libraries_sha256'])
 files.extend(ROOT/n for n in ('src/ve_sd_turbo_cont_transpose.c','tests/benchmark_sd_cont_transpose.cpp','scripts/benchmark_sd_cont_transpose.sh','scripts/prepare_sd_cont_transpose_fixtures.py','build/sd-cont-transpose-probe/candidate.o','build/sd-cont-transpose-probe/probe','build/sd-cont-transpose-symbols.log','build/sd-cont-transpose-relocations.log','build/sd-cont-transpose-fixture-rejection.log','build/sd-cont-transpose-audit-thread-log-rejection.log'))
 diagnostics=[dict(line=int(n),message=s.strip()) for n,s in re.findall(r'\bve_sd_turbo_cont_transpose\.c, line (\d+): ([^\n]+)',log)]
 report=dict(status='cont_transpose_microbenchmark_verified',baseline=binding,results=results,checks=Counter(r[0] for r in checks),cpu_bit_checks=6,timed_graphs=72,actual_graph_correctness_cases=24,actual_thread_counts=[1,2,4,8],compiler_diagnostics=diagnostics,candidate_effective_optimization='-O2',model_unchanged=True,temperature=thermal(guard),fan_detail=fan_detail(guard),artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_abba.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py')},scope='independent candidate O2 pure transpose, actual model O1 ggml CONT graph baseline versus custom candidate graph, common actual runtime; three measured layouts, 12 samples/mode/shape with ABBA; full bit-pattern/direct/concurrent checks; graph overhead included, no model integration or full-model speedup claim')
 safe(report);target.write_text(json.dumps(report,indent=2)+'\n');print('CONT transpose microbenchmark audited:',target.relative_to(ROOT));print([(r['shape'],r['baseline_seconds'],r['candidate_seconds'],r['latency_reduction_percent']) for r in results])
if __name__=='__main__':main()
