"""Publish checked SD-Turbo im2col timings, scope and thermal evidence."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import time
from record_qwen36_mtp import safe,thermal

ROOT=Path(__file__).resolve().parents[1]

def fan_detail(log):
    matches=re.findall(r'Fan observation:.*log=([^\s]+)',log.read_text())
    if not matches:raise RuntimeError('fan observation log required')
    path=(ROOT/matches[-1]).resolve();path.relative_to(ROOT/'build')
    rpm={}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if re.search(r'/fan\d+_input$',row['channel']):
                rpm.setdefault(row['channel'],set()).add(row['raw_value'])
    values=[float(v) for items in rpm.values() for v in items if v.isdigit()]
    return {'sysfs_rpm_channels':len(rpm),'sysfs_rpm_channels_with_changes':sum(len(v)>1 for v in rpm.values()),
            'sysfs_rpm_aggregate_range':[min(values),max(values)] if values else None,
            'all_observed_rpm_values_zero':bool(values) and all(v==0 for v in values),
            'interpretation':'zero sysfs readings do not establish physical fan speed; BMC availability is recorded separately; actual fan policy not verified'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--benchmark-log',type=Path,required=True)
    parser.add_argument('--four-step',type=Path,required=True)
    parser.add_argument('--four-step-log',type=Path,required=True)
    parser.add_argument('--build-log',type=Path,required=True)
    parser.add_argument('--operators-log',type=Path,required=True)
    parser.add_argument('--unit-log',type=Path,required=True)
    parser.add_argument('--graph-log',type=Path,required=True)
    parser.add_argument('--six-case',type=Path,required=True)
    parser.add_argument('--six-case-log',type=Path,required=True)
    args=parser.parse_args()
    bench=json.loads((args.benchmark/'summary.json').read_text())
    four=json.loads((args.four_step/'summary.json').read_text())
    manifest=json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    binary=hashlib.sha256((ROOT/'build/sd-baseline-ve/bin/sd').read_bytes()).hexdigest()
    if (bench.get('kind')!='sd_turbo_im2col_abba' or bench.get('fixed_silu_mode')!='ve' or bench.get('fixed_softmax_mode')!='ve' or bench.get('mode_axis')!='im2col' or not bench['completed'] or len(bench['runs'])!=4 or not bench['all_numerical_checks_passed'] or
            [r['mode'] for r in bench['runs']]!=['generic','ve','ve','generic'] or
            bench['binary_sha256']!=binary):
        raise RuntimeError('matching completed ABBA required')
    if (not four['completed'] or four['steps']!=4 or four['threadpool_mode']!='persistent' or four.get('silu_mode')!='ve' or four.get('softmax_mode')!='ve' or four.get('im2col_mode')!='ve' or four.get('operator_profile_enabled',False) or
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
    unit=args.unit_log.read_text()
    if unit.count('independent_bitwise_tail PASS')!=16:
        raise RuntimeError('16 independent bitwise and tail fixtures required')
    if not re.search(r've_sd_turbo_im2col\.c, line \d+: Vectorized loop\.',args.build_log.read_text()):
        raise RuntimeError('compiler vectorization diagnostics required')
    # Preserve an exact comparison as additional evidence, without tightening
    # the planned numerical tolerances or publishing any raw image arrays.
    first=ROOT/bench['runs'][0]['artifacts']/'case0'
    names=sorted(p.name for p in first.glob('*.f32'))
    if len(names)!=7:raise RuntimeError('seven single-step traces required')
    equality={name:all(hashlib.sha256((ROOT/run['artifacts']/'case0'/name).read_bytes()).digest()==
                       hashlib.sha256((first/name).read_bytes()).digest() for run in bench['runs'])
              for name in names+['image.png']}
    if not all(equality[name] for name in names):
        raise RuntimeError('data-only candidate must preserve every native trace bit')
    if 'IM2COL_DISPATCH_PASS optimized=1 fallback=2' not in args.graph_log.read_text():
        raise RuntimeError('actual optimized and fallback dispatch validation required')
    six=json.loads((args.six_case/'summary.json').read_text())
    if (not six['completed'] or six['steps']!=1 or len(six['cases'])!=6 or six['binary_sha256']!=binary
            or six['test_sha256']!=bench['checker_sha256'] or six.get('im2col_mode')!='ve'
            or six.get('silu_mode')!='ve' or six.get('softmax_mode')!='ve' or six['threadpool_mode']!='persistent'
            or six['threads']!=bench['threads'] or not all(len(c['checks'])==5 and all(x['passed'] for x in c['checks']) for c in six['cases'])):
        raise RuntimeError('six complete same-binary candidate cases required')
    eligible=bench['decision']=='faster_pending_four_step_validation'
    def relative(path):
        value=path.resolve();value.relative_to(ROOT/'build');return str(value.relative_to(ROOT))
    report={'status':'im2col_candidate_verified' if eligible else 'im2col_candidate_not_adopted',
            'recommended_for_validated_workload':eligible,'default_enabled':False,
            'model_repo':'stabilityai/sd-turbo','model_revision':four['model_revision'],
            'runtime_slot':1,'precision':'FP32','size':512,
            'implementation':f"isolated vector F32 im2col data movement; SiLU and Softmax ve; persistent pools, {bench['threads']}-thread comparison, poll=0; NLC unchanged",
            'scope':'one fixed single-step prompt, two ABBA runs per arm; one four-step candidate and six single-step numerical cases; no general workload or resident steady-state claim',
            'physical_hbm_peak_measured':False,'quality_benchmark_performed':False,
            'build':manifest,'build_temperature':thermal(args.build_log),
            'operators_temperature':thermal(args.operators_log),
            'im2col_unit_temperature':thermal(args.unit_log),
            'im2col_unit_cases':16,'im2col_unit_max_abs_error':0.0,
            'compiler_vectorization_confirmed':True,
            'unit_sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path('tests/check_sd_im2col.c'),Path('scripts/check_sd_im2col.sh'),Path('tests/check_sd_im2col_graph.cpp'),Path('scripts/check_sd_im2col_graph.sh'))},
            'previous_silu_range_fix_retained':True,
            'benchmark_artifacts':relative(args.benchmark),'benchmark':bench,
            'abba_native_artifact_bytewise_equality':equality,
            'remaining_abba_temperature':thermal(args.benchmark_log),
            'abba_fan_observation_detail':fan_detail(args.benchmark_log),
            'four_step_artifacts':relative(args.four_step),'four_step':four,
            'four_step_temperature':thermal(args.four_step_log),
            'graph_dispatch_temperature':thermal(args.graph_log),
            'six_case_artifacts':relative(args.six_case),'six_case':six,
            'six_case_temperature':thermal(args.six_case_log)}
    safe(report)
    out=ROOT/'docs/results'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-sd-turbo-im2col.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Public im2col evidence:',out.relative_to(ROOT))

if __name__=='__main__':main()
