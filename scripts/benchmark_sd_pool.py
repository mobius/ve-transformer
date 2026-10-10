"""Guarded, correctness-checked ABBA of one fixed SD-Turbo prompt.

The first baseline is a separately guarded profiling run, supplied with its
completed temperature log. Remaining runs execute serially under this guard.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from record_qwen36_mtp import thermal

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--baseline-log',type=Path,required=True)
    parser.add_argument('--threads',type=int,choices=(1,2,4,8),default=8)
    args=parser.parse_args()
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':
        raise RuntimeError('temperature supervision required')
    reference=args.reference.resolve();reference.relative_to(ROOT/'build')
    baseline=args.baseline.resolve();baseline.relative_to(ROOT/'build')
    reference_data=json.loads((reference/'summary.json').read_text())
    if not reference_data['completed'] or reference_data['steps']!=1 or len(reference_data['cases'])!=1:
        raise RuntimeError('one completed single-step CPU reference required')
    checker=ROOT/'tests/check_sd_turbo.py'
    executable=ROOT/'build/sd-baseline-ve/bin/sd'
    checker_sha=sha(checker);binary_sha=sha(executable)
    baseline_temperature=thermal(args.baseline_log)
    if any(value>=85 if sensor=='cpu' else value>=75
           for sensor,value in baseline_temperature['peaks_c'].items()):
        raise RuntimeError('baseline exceeded temperature policy')
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-pool-abba',time.gmtime())
    folder.mkdir()
    report={'completed':False,'adopted':False,'kind':'sd_turbo_pool_abba','steps':1,'threads':args.threads,
            'reference_artifacts':str(reference.relative_to(ROOT)),
            'binary_sha256':binary_sha,'checker_sha256':checker_sha,
            'benchmark_sha256':sha(Path(__file__)),'first_baseline_temperature':baseline_temperature,
            'scope':'one fixed prompt; fresh-process component loading and diagnostic traces; filesystem cache not controlled; two runs per arm; no resident steady-state claim',
            'profile_scope':'nested timings: graph compute includes copies/synchronization; backend timings cover synchronous native VE/NLC calls',
            'runs':[]}
    def save():
        (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save();(folder/'benchmark_sd_pool.py').write_bytes(Path(__file__).read_bytes())
    def collect(path,mode):
        value=json.loads((path/'summary.json').read_text())
        if (not value['completed'] or len(value['cases'])!=1 or value['steps']!=1 or
                value['threads']!=args.threads or value['threadpool_mode']!=mode or not value['profile_enabled'] or
                value['binary_sha256']!=binary_sha or value['test_sha256']!=checker_sha or
                value['reference_artifacts']!=str(reference.relative_to(ROOT)) or
                sha(path/'check_sd_turbo.py')!=checker_sha):
            raise RuntimeError('ABBA execution provenance differs')
        case=value['cases'][0]
        if case['returncode'] or len(case['checks'])!=5 or not all(c['passed'] for c in case['checks']):
            raise RuntimeError('five completed numerical checks required per run')
        metrics={part:sum(x['seconds'] for x in case['profile'] if x['part']==part)
                 for part in ('graph_build','graph_alloc','graph_compute','weights_load','backend_CPU','backend_BLAS')}
        metrics['process_seconds']=case['process_seconds']
        total=[x['seconds'] for x in case['profile'] if x['stage']=='pipeline' and x['part']=='total']
        if len(total)!=1:raise RuntimeError('one native pipeline total required')
        metrics['pipeline_total']=total[0]
        report['runs'].append({'mode':mode,'artifacts':str(path.relative_to(ROOT)),
                               'metrics_seconds':metrics,'validation':value})
        save()
    collect(baseline,'disposable')
    for index,mode in enumerate(('persistent','persistent','disposable'),1):
        if sha(checker)!=checker_sha or sha(executable)!=binary_sha:
            raise RuntimeError('checker or binary changed during ABBA')
        print('ABBA run',index,mode,flush=True)
        command=[sys.executable,str(checker),'--framework','compact','--reference',str(reference),
                 '--threads',str(args.threads),'--threadpool',mode,'--profile']
        log=folder/('run%d-checker.log'%index)
        with log.open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=7200)
        if result.returncode:raise RuntimeError('ABBA image or numerical validation failed; inspect private log')
        matched=re.findall(r'Native full image stages match CPU reference: (build/results/[^\s]+)',log.read_text())
        if len(matched)!=1:raise RuntimeError('one completed native artifact required')
        path=(ROOT/matched[0]).resolve();path.relative_to(ROOT/'build')
        collect(path,mode)
        print('ABBA run complete',index,mode,report['runs'][-1]['metrics_seconds'],flush=True)
    groups={mode:[r['metrics_seconds'] for r in report['runs'] if r['mode']==mode]
            for mode in ('disposable','persistent')}
    comparisons={}
    for part in groups['disposable'][0]:
        a=[r[part] for r in groups['disposable']];b=[r[part] for r in groups['persistent']]
        base=statistics.mean(a);candidate=statistics.mean(b)
        comparisons[part]={'baseline_mean_seconds':base,'candidate_mean_seconds':candidate,
                           'baseline_range_seconds':[min(a),max(a)],'candidate_range_seconds':[min(b),max(b)],
                           'latency_reduction_percent':100*(1-candidate/base) if base else None,
                           'speedup_percent':100*(base/candidate-1) if candidate else None}
    report['comparisons']=comparisons
    faster=all(comparisons[part]['latency_reduction_percent']>2 and
               comparisons[part]['candidate_range_seconds'][1]<comparisons[part]['baseline_range_seconds'][0]
               for part in ('graph_compute','process_seconds'))
    report.update(completed=True,decision='faster_pending_four_step_validation' if faster else 'not_adopted',
                  all_numerical_checks_passed=True)
    save();print('ABBA complete:',folder.relative_to(ROOT),'decision:',report['decision'],flush=True)

if __name__=='__main__':main()
