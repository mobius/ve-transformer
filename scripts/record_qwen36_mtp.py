"""Publish aggregate MTP evidence without prompts, generated text or token IDs."""
import argparse
import csv
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def safe(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in {'text','greedy','greedy_tokens','tokens','prompt','token_ids'}:
                raise RuntimeError('private sequence field in aggregate report')
            safe(item)
    elif isinstance(value,list):
        for item in value:safe(item)
    elif isinstance(value,str):
        if re.search(r'/ho[m]e/|/mnt/stor[a]ge/',value):raise RuntimeError('private path in report')

def thermal(path):
    text=path.read_text()
    lines=[line for line in text.splitlines() if line.startswith('Temperature summary:')]
    if not lines:raise RuntimeError('completed temperature summary required')
    fan_paths=re.findall(r'Fan observation:.*log=([^\s]+)',text)
    statuses=set()
    if fan_paths:
        fan_path=(ROOT/fan_paths[-1]).resolve();fan_path.relative_to(ROOT)
        with fan_path.open() as handle:
            for row in csv.DictReader(handle):
                if row['channel']=='bmc/read_status':statuses.add(row['raw_value'])
    return {'peaks_c':{key:float(value) for key,value in re.findall(r'(cpu|ve[0-9]+) peak=([0-9.]+)C',lines[-1])},
            'fan_channels':int(re.findall(r'Fan observation: ([0-9]+) channels',text)[-1]),
            'observed_fan_changes':int(re.findall(r', ([0-9]+) observed changes',text)[-1]),
            'bmc_read_statuses_observed':sorted(statuses),'fan_policy_conclusion':'no observed changes in readable channels; actual policy stability not established'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--validation-log',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path)
    parser.add_argument('--benchmark-log',type=Path)
    parser.add_argument('--depth-screen',type=Path)
    parser.add_argument('--depth-log',type=Path)
    args=parser.parse_args()
    validation=json.loads((args.validation/'summary.json').read_text())
    if not validation['completed'] or len(validation['cases'])!=3 or any(c['greedy_agreement']!=12 for c in validation['cases']):
        raise RuntimeError('three completed independent CPU greedy cases required')
    if 'perturbed_rollback' not in validation or 'natural_end' not in validation or 'prefill_boundary' not in validation:
        raise RuntimeError('rollback, natural end and prefill boundary acceptance required')
    report={'status':'native_mtp_ve_verified','default_adopted':False,'runtime_slot':1,
            'validation_artifacts':str(args.validation.resolve().relative_to(ROOT)),
            'validation':validation,'validation_temperature':thermal(args.validation_log),
            'serving':'isolated CLI and repeated requests in one process; persistent external protocol not implemented',
            'scope':'greedy text inference; no sampling, vision, other VE cards or maximum trained context coverage',
            'physical_hbm_peak_measured':False,
            'rejected_iterations':['docs/results/20261007T014420Z-qwen36-native-mtp-state-api-rejected.json'],
            'compiler_temperature_event':{'unlimited_build_peak_cpu_c':87,'stopped':True,'limited_build_peak_cpu_c':78,'compiler_duty_percent':25}}
    log=(args.validation/'case0-mtp.stderr').read_text()
    report['logged_backend_buffer_requests_mib']={kind:[float(value) for value in re.findall(r'CPU '+kind+r' buffer size =\s*([0-9.]+) MiB',log)]
                                                  for kind in ('model','KV','RS','compute','output')}
    report['backend_buffer_scope']='Native VE CPU backend allocation requests; repeated resizing entries and other allocator/NLC overhead excluded; not physical HBM peak'
    if args.benchmark:
        if args.benchmark_log is None:raise RuntimeError('benchmark temperature log required')
        benchmark=json.loads((args.benchmark/'summary.json').read_text())
        if not benchmark['completed'] or benchmark['kind']!='benchmark' or len(benchmark['abba_runs'])!=4:
            raise RuntimeError('completed ABBA measurement required')
        for key in ('binary_sha256','target_sha256','mtp_sha256'):
            if benchmark[key]!=validation[key]:raise RuntimeError('benchmark artifact differs from validation')
        report.update(benchmark=benchmark,benchmark_temperature=thermal(args.benchmark_log),
                      benchmark_artifacts=str(args.benchmark.resolve().relative_to(ROOT)))
    if args.depth_screen:
        if args.depth_log is None:raise RuntimeError('depth temperature log required')
        depth=json.loads((args.depth_screen/'summary.json').read_text())
        if not depth['completed'] or depth['kind']!='depth_screen' or depth['draft_tokens']!=1:
            raise RuntimeError('completed one-draft screening required')
        for key in ('binary_sha256','target_sha256','mtp_sha256'):
            if depth[key]!=validation[key]:raise RuntimeError('depth artifact differs from validation')
        report.update(depth_screen=depth,depth_temperature=thermal(args.depth_log),
                      depth_artifacts=str(args.depth_screen.resolve().relative_to(ROOT)))
    safe(report)
    output=ROOT/'docs/results/20261007T011801Z-qwen36-native-mtp.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Public MTP evidence ready:',output.relative_to(ROOT))

if __name__=='__main__':main()
