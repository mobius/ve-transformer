"""Audit complete same-binary im2col ABBA, including negative results."""
import argparse
import csv
import json
from pathlib import Path
import re
from record_sd_gelu_result import check_requests
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail

ROOT = Path(__file__).resolve().parents[1]


def rows_dispatch(folder, value, mode):
    text = (folder/'native.log').read_text()
    segments = re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+', text, re.S)
    for request, body in zip(value['requests'], segments):
        actual = [dict(stage=a, rows=int(b), channels=int(c), enabled=int(d))
                  for a,b,c,d in re.findall(r'SD_IM2COL_ROWS stage=(clip|unet|vae) rows=(\d+) channels=(\d+) enabled=(0|1)',body)]
        if actual != request['im2col_row_dispatch']:
            raise RuntimeError('native im2col dispatch differs')
        for stage, total in (('clip',0),('unet',98),('vae',40)):
            entries = [r for r in actual if r['stage']==stage]
            wanted = (15 if mode=='rows_256' else 8) if mode!='channels' and stage=='vae' else 0
            if len(entries)!=(value['steps'] if stage=='unet' else 1) or any(
                    (r['rows'],r['channels'],r['enabled'])!=(wanted,total-wanted,int(mode!='channels')) for r in entries):
                raise RuntimeError('candidate coverage differs')
        scopes=[dict(stage=a,extended=int(b)) for a,b in re.findall(r'SD_IM2COL_SCOPE stage=(clip|unet|vae) extended=(0|1)',body)]
        if mode=='rows_256' and not scopes:
            raise RuntimeError('missing extended scope evidence')
        if scopes and (scopes!=request.get('im2col_scope') or [r['stage'] for r in scopes]!=[r['stage'] for r in actual] or any(r['extended']!=int(mode=='rows_256') for r in scopes)):
            raise RuntimeError('scope evidence differs')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--guard-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-proof',type=Path,default=ROOT/'docs/results/20261009T130018Z-sd-turbo-im2col-model.json')
    args=parser.parse_args()
    folder=args.benchmark.resolve();folder.relative_to(ROOT/'build')
    bench=json.loads((folder/'summary.json').read_text())
    extended=bench['kind']=='sd_turbo_im2col_extend_abba'
    baseline_mode,candidate_mode=('rows','rows_256') if extended else ('channels','rows')
    if not bench['completed'] or bench['kind'] not in ('sd_turbo_im2col_abba','sd_turbo_im2col_extend_abba') or [r['mode'] for r in bench['runs']]!=[baseline_mode,candidate_mode,candidate_mode,baseline_mode]:
        raise RuntimeError('complete ordered im2col comparison required')
    if sha(folder/'benchmark_sd_resident.py')!=bench['benchmark_sha256']:
        raise RuntimeError('frozen benchmark differs')
    manifest_path=ROOT/'build/sd-baseline-ve/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('model build evidence changed')
    if manifest['sha256']['build/sd-baseline-ve/bin/sd']!=bench['binary_sha256']:
        raise RuntimeError('model binary differs')
    prior_path=args.prior_proof.resolve();prior_path.relative_to(ROOT/'docs/results')
    prior=json.loads(prior_path.read_text());earlier=0
    for row in prior['tests']:
        for name,digest in row['sha256'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('prior model evidence changed')
        native=ROOT/row['artifacts'];value=json.loads((native/'summary.json').read_text())
        earlier+=check_requests(native,value,bench['binary_sha256'],bench['checker_sha256'])
        rows_dispatch(native,value,prior.get('im2col_mode','rows'))
    if earlier!=62 or prior['independent_cpu_checks']!=earlier:
        raise RuntimeError('62 prior model checks required')
    checks=0;evidence={}
    for index,run in enumerate(bench['runs']):
        native=ROOT/run['artifacts'];sampled=ROOT/run['memory_artifacts']
        value=json.loads((native/'summary.json').read_text())
        if 'SD_IM2COL_SHAPE ' in (native/'native.log').read_text():raise RuntimeError('shape diagnostics enabled in timing comparison')
        if value['steps']!=1 or len(value['requests'])!=2 or value['reference_artifacts']!=bench['reference_artifacts']:
            raise RuntimeError('fixed reference differs')
        if (value['gelu_mode']!='ve' or value['binary_scalar_mode']!='ve' or value['vae_threads']!=8 or value['vae_blas_threads']!=4 or
            value['operator_profile_enabled'] or value['binary_shape_profile_enabled'] or value['nlc_threads']!='unified' or value['mode']!='resident' or value['tokenizer_mode']!='resident'):
            raise RuntimeError('comparison configuration differs')
        checks+=check_requests(native,value,bench['binary_sha256'],bench['checker_sha256'])
        rows_dispatch(native,value,run['mode'])
        if value['requests']!=run['requests'] or value['process_seconds']!=run['process_seconds']:
            raise RuntimeError('aggregate differs from raw result')
        memory=json.loads((sampled/'summary.json').read_text())
        csv_rows=list(csv.DictReader((sampled/'memory.csv').open()))
        if not memory['completed'] or memory!=run['memory'] or memory['final_used_kib']!=131072 or max(int(r['used_kib']) for r in csv_rows)!=memory['sampled_highest_used_kib'] or int(csv_rows[-1]['used_kib'])!=131072:
            raise RuntimeError('memory recovery/sample mismatch')
        files=[native/'summary.json',native/'native.log',native/'check_sd_resident.py',sampled/'summary.json',sampled/'memory.csv',folder/('run%d.log'%index)]
        evidence[str(index)]={str(p.relative_to(ROOT)):sha(p) for p in files}
    if checks!=40:raise RuntimeError('40 comparison checks required')
    equality=all(r['requests'][i]['trace_sha256']==bench['runs'][0]['requests'][i]['trace_sha256'] and r['requests'][i]['png_sha256']==bench['runs'][0]['requests'][i]['png_sha256'] for r in bench['runs'] for i in (0,1))
    if not equality or not bench['all_trace_and_png_bytes_identical']:
        raise RuntimeError('data movement must preserve trace/PNG bytes')
    metrics={}
    for key,index in [('process',None),('request0',0),('request1',1)]:
        mean={mode:sum(r['process_seconds'] if index is None else r['requests'][index]['request_seconds'] for r in bench['runs'] if r['mode']==mode)/2 for mode in (baseline_mode,candidate_mode)}
        baseline_key,candidate_key=('rows512_seconds','rows256_seconds') if extended else ('channels_seconds','rows_seconds')
        metrics[key]={baseline_key:mean[baseline_mode],candidate_key:mean[candidate_mode],'latency_reduction_percent':100*(1-mean[candidate_mode]/mean[baseline_mode])}
    if metrics!=bench['metrics']:raise RuntimeError('metric mismatch')
    stage_means={stage:{mode:{part:sum(sum(p['seconds'] for p in r['requests'][1]['profile'] if p['stage']==stage and p['part']==part) for r in bench['runs'] if r['mode']==mode)/2 for part in ('backend_CPU','backend_BLAS','graph_compute')} for mode in (baseline_mode,candidate_mode)} for stage in ('clip','unet','vae')}
    target=ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir'
    flags=(target/'flags.make').read_text()
    global_flags=re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1]
    make=(target/'build.make').read_text()
    optimization={}
    object_sha={}
    for name in ('ve_sd_turbo_im2col.c','ve_sd_turbo_im2col_rows.c'):
        command=next(line for line in make.splitlines() if '$(C_FLAGS)' in line and line.endswith('-c '+str(ROOT/'src'/name)))
        options=global_flags+' '+command.split('$(C_FLAGS)',1)[1].split('-MD',1)[0]
        optimization[name]=re.findall(r'(?<!\S)-O[0-3](?!\S)',options)[-1]
        objects=list(target.rglob(name+'.o'))
        if len(objects)!=1:raise RuntimeError('unique model kernel object required')
        object_sha[name]=sha(objects[0])
    if 'actual_model_object_sha256' in prior and object_sha['ve_sd_turbo_im2col_rows.c']!=prior['actual_model_object_sha256']:
        raise RuntimeError('current candidate differs from independently checked model object')
    if 'model_manifest_sha256' in prior and sha(manifest_path)!=prior['model_manifest_sha256']:
        raise RuntimeError('candidate proof refers to another build')
    guard=args.guard_log.read_text()
    traces=re.findall(r'log=(build/results/[^\s]+\.csv)',guard)
    if len(traces)!=2:raise RuntimeError('temperature and fan CSV paths required')
    temperature_csv=ROOT/traces[0]
    samples=list(csv.DictReader(temperature_csv.open()))
    if not samples or any(not float(r['temperature_c'])<float(r['stop_c']) for r in samples):
        raise RuntimeError('missing or unsafe temperature samples')
    report=dict(comparison=bench["comparison"],metrics=metrics,hot_stage_profile_mean_seconds=stage_means,
                decision='faster_in_fixed_workload' if metrics['request1']['latency_reduction_percent']>0 else 'not_faster',
                default_enabled=False,same_binary=True,model_kernel_optimization=optimization,model_kernel_object_sha256=object_sha,compile_metadata_sha256={"flags.make":sha(target/"flags.make"),"build.make":sha(target/"build.make")},all_trace_and_png_bytes_identical=equality,
                binary_sha256=bench['binary_sha256'],checker_sha256=bench['checker_sha256'],benchmark_sha256=bench['benchmark_sha256'],publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={name:sha(ROOT/name) for name in ("scripts/record_sd_gelu_result.py","scripts/record_qwen36_mtp.py","scripts/record_sd_im2col_result.py","scripts/benchmark_sd_runtime.py")},
                earlier_proof_sha256=sha(prior_path),earlier_proof_path=str(prior_path.relative_to(ROOT)),benchmark_summary_sha256=sha(folder/'summary.json'),build_manifest_sha256=sha(manifest_path),model_object_ownership_groups=prior.get('model_object_ownership_groups'),model_object_distinct_concurrent_groups=prior.get('model_object_distinct_concurrent_groups'),earlier_cpu_stage_checks=earlier,abba_cpu_stage_checks=checks,total_cpu_stage_checks=earlier+checks,
                temperature=thermal(args.guard_log),fan_detail=fan_detail(args.guard_log),warning_events=guard.count('Temperature warning:'),guard_log_sha256=sha(args.guard_log),thermal_artifact_sha256={name:sha(ROOT/name) for name in traces},artifact_sha256=evidence,
                runs=[dict(mode=r['mode'],process_seconds=r['process_seconds'],request_seconds=[q['request_seconds'] for q in r['requests']],sampled_node_peak_gib=r['memory']['sampled_highest_used_kib']/1048576,final_node_used_mib=128) for r in bench['runs']],
                scope='single VE, fixed FP32 512 one-step prompt, two runs/mode; cold caches uncontrolled; discrete whole-node memory; no arbitrary production/concurrency claim')
    safe(report);output=args.output.resolve();output.relative_to(ROOT/'docs/results');output.write_text(json.dumps(report,indent=2)+'\n')
    print('Audited im2col ABBA:',json.dumps(metrics))


if __name__=='__main__':main()
