"""Add a completed candidate test to a frozen-model proof, without old-build checks."""
import argparse
import csv
import json
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe, thermal
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--name',choices=('double','six','four'),required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--memory',type=Path,required=True)
    parser.add_argument('--guard-log',type=Path,required=True)
    args=parser.parse_args()
    proof=args.proof.resolve();proof.relative_to(ROOT/'docs/results')
    run=args.run.resolve();run.relative_to(ROOT/'build')
    memory=args.memory.resolve();memory.relative_to(ROOT/'build')
    value=json.loads(proof.read_text())
    manifest=json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    if sha(ROOT/'build/sd-baseline-ve/manifest.json')!=value['model_manifest_sha256']:
        raise RuntimeError('model changed since actual-object validation')
    for name,digest in manifest['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('model source/binary changed')
    objects=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('ve_sd_turbo_im2col_rows.c.o'))
    if len(objects)!=1 or sha(objects[0])!=value['actual_model_object_sha256']:
        raise RuntimeError('model object differs from independently checked object')
    result=json.loads((run/'summary.json').read_text())
    steps,sequence={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[args.name]
    if result['steps']!=steps or [r['reference_case'] for r in result['requests']]!=sequence:
        raise RuntimeError('wrong workload')
    if (result['gelu_mode']!='ve' or result['binary_scalar_mode']!='ve' or result['vae_threads']!=8 or result['vae_blas_threads']!=4 or result['nlc_threads']!='unified' or result['tokenizer_mode']!='resident' or result['mode']!='resident'):
        raise RuntimeError('candidate configuration differs')
    checks=check_requests(run,result,manifest['sha256']['build/sd-baseline-ve/bin/sd'])
    candidate_mode=value.get('im2col_mode','rows')
    if result.get('im2col_mode',candidate_mode)!=candidate_mode:raise RuntimeError('candidate mode differs')
    rows_dispatch(run,result,candidate_mode)
    sampled=json.loads((memory/'summary.json').read_text())
    rows=list(csv.DictReader((memory/'memory.csv').open()))
    if not sampled['completed'] or sampled['returncode'] or sampled['final_used_kib']!=131072 or int(rows[-1]['used_kib'])!=131072 or max(int(r['used_kib']) for r in rows)!=sampled['sampled_highest_used_kib']:
        raise RuntimeError('memory completion/recovery mismatch')
    observed=thermal(args.guard_log)
    paths=re.findall(r'log=(build/results/[^\s]+\.csv)',args.guard_log.read_text())
    if len(paths)!=2:raise RuntimeError('thermal/fan CSVs required')
    samples=list(csv.DictReader((ROOT/paths[0]).open()))
    if not samples or any(not float(r['temperature_c'])<float(r['stop_c']) for r in samples):
        raise RuntimeError('unsafe/missing temperature samples')
    files=[run/'summary.json',run/'native.log',memory/'summary.json',memory/'memory.csv',args.guard_log.resolve()]+[ROOT/p for p in paths]
    entry=dict(name=args.name,artifacts=str(run.relative_to(ROOT)),memory_artifacts=str(memory.relative_to(ROOT)),guard_log=str(args.guard_log.resolve().relative_to(ROOT)),
               sha256={str(p.relative_to(ROOT)):sha(p) for p in files},request_seconds=[r['request_seconds'] for r in result['requests']],cpu_checks=checks,temperature=observed)
    existing=[r for r in value.get('tests',[]) if r['name']!=args.name]
    for row in existing:
        for name,digest in row['sha256'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('earlier candidate evidence changed')
    value['tests']=sorted(existing+[entry],key=lambda r:('double','six','four').index(r['name']))
    value['completed_tests']=[r['name'] for r in value['tests']]
    value['independent_cpu_checks']=sum(r['cpu_checks'] for r in value['tests'])
    value[args.name]=entry
    value['all_candidate_tests_completed']=value['completed_tests']==['double','six','four']
    value['scope']='O2 '+candidate_mode+' model candidate correctness; controlled performance comparison separate'
    value['performance_measured']=True
    value['request_seconds']=value['tests'][0]['request_seconds']
    value['publisher_sha256']=sha(Path(__file__))
    safe(value);proof.write_text(json.dumps(value,indent=2)+'\n')
    print('Verified candidate tests:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])


if __name__=='__main__':main()
