"""Publish checked SD-Turbo pool timings, scope and thermal evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from record_qwen36_mtp import safe,thermal

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--benchmark-log',type=Path,required=True)
    parser.add_argument('--four-step',type=Path,required=True)
    parser.add_argument('--four-step-log',type=Path,required=True)
    parser.add_argument('--build-log',type=Path,required=True)
    parser.add_argument('--operators-log',type=Path,required=True)
    args=parser.parse_args()
    bench=json.loads((args.benchmark/'summary.json').read_text())
    four=json.loads((args.four_step/'summary.json').read_text())
    manifest=json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    binary=hashlib.sha256((ROOT/'build/sd-baseline-ve/bin/sd').read_bytes()).hexdigest()
    if (not bench['completed'] or len(bench['runs'])!=4 or not bench['all_numerical_checks_passed'] or
            [r['mode'] for r in bench['runs']]!=['disposable','persistent','persistent','disposable'] or
            bench['binary_sha256']!=binary):
        raise RuntimeError('matching completed ABBA required')
    if (not four['completed'] or four['steps']!=4 or four['threadpool_mode']!='persistent' or
            not four['profile_enabled'] or len(four['cases'])!=1 or four['binary_sha256']!=binary or
            four['test_sha256']!=bench['checker_sha256'] or four['threads']!=bench['threads']):
        raise RuntimeError('matching completed four-step candidate required')
    case=four['cases'][0]
    if case['returncode'] or len(case['checks'])!=11 or not all(c['passed'] for c in case['checks']):
        raise RuntimeError('eleven passed four-step checks required')
    if manifest['backend']!='ve' or manifest['sha256']['build/sd-baseline-ve/bin/sd']!=binary:
        raise RuntimeError('native VE build required')
    if 'IMAGE_OPERATOR_PASS cases=12;' not in args.operators_log.read_text():
        raise RuntimeError('current twelve-operator regression required')
    # Preserve an exact comparison as additional evidence, without tightening
    # the planned numerical tolerances or publishing any raw image arrays.
    first=ROOT/bench['runs'][0]['artifacts']/'case0'
    names=sorted(p.name for p in first.glob('*.f32'))
    if len(names)!=7:raise RuntimeError('seven single-step traces required')
    equality={name:all(hashlib.sha256((ROOT/run['artifacts']/'case0'/name).read_bytes()).digest()==
                       hashlib.sha256((first/name).read_bytes()).digest() for run in bench['runs'])
              for name in names+['image.png']}
    eligible=bench['decision']=='faster_pending_four_step_validation'
    def relative(path):
        value=path.resolve();value.relative_to(ROOT/'build');return str(value.relative_to(ROOT))
    report={'status':'pool_candidate_verified' if eligible else 'pool_candidate_not_adopted',
            'recommended_for_validated_workload':eligible,'default_enabled':False,
            'model_repo':'stabilityai/sd-turbo','model_revision':four['model_revision'],
            'runtime_slot':1,'precision':'FP32','size':512,
            'implementation':f"one native VE pool per component, {bench['threads']}-thread comparison, poll=0; NLC unchanged",
            'scope':'one fixed single-step prompt, two ABBA runs per arm; one four-step candidate; no general workload or resident steady-state claim',
            'physical_hbm_peak_measured':False,'quality_benchmark_performed':False,
            'build':manifest,'build_temperature':thermal(args.build_log),
            'operators_temperature':thermal(args.operators_log),
            'benchmark_artifacts':relative(args.benchmark),'benchmark':bench,
            'abba_native_artifact_bytewise_equality':equality,
            'remaining_abba_temperature':thermal(args.benchmark_log),
            'four_step_artifacts':relative(args.four_step),'four_step':four,
            'four_step_temperature':thermal(args.four_step_log)}
    safe(report)
    out=ROOT/'docs/results'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-sd-turbo-pool.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Public pool evidence:',out.relative_to(ROOT))

if __name__=='__main__':main()
