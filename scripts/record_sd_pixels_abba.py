"""Audit same-binary pixels ABBA against current model checks."""
import argparse
import json
import re
from pathlib import Path
from record_sd_pixels_model import audited_pixels_run,pixels_compiler_evidence
from record_sd_png_model import png_compiler_evidence
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 folder=a.benchmark.resolve();folder.relative_to(ROOT/'build');guard=a.guard_log.resolve();guard.relative_to(ROOT/'build');proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results')
 value=json.loads((folder/'summary.json').read_text());prior=json.loads(proof.read_text())
 if (not value['completed'] or value['kind']!='sd_turbo_pixels_abba' or value['comparison']!='pixels' or value['case_sequence']!='0,0,0,0' or [r['mode'] for r in value['runs']]!=['generic','ve','ve','generic'] or not prior['all_candidate_tests_completed'] or prior['independent_cpu_checks']!=62 or prior['completed_tests']!=['double','six','four'] or value['binary_sha256']!=prior['binary_sha256'] or value['checker_sha256']!=prior['checker_sha256']):raise RuntimeError('complete matching formal and prior model evidence required')
 if sha(folder/'benchmark_sd_resident.py')!=value['benchmark_sha256']:raise RuntimeError('frozen benchmark changed')
 prior_checks=0
 for entry in prior['tests']:
  for name,digest in entry['sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError('prior model evidence changed')
  logs=[ROOT/n for n in entry['sha256'] if n.endswith('.log') and not n.endswith('native.log')]
  if len(logs)!=1:raise RuntimeError('prior guard log required')
  native,audited,manifest=audited_pixels_run(ROOT/entry['artifacts'],ROOT/entry['memory_artifacts'],logs[0],'ve')
  steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[entry['name']]
  if native['steps']!=steps or [r['reference_case'] for r in native['requests']]!=cases or manifest!=prior['model_manifest_sha256']:raise RuntimeError('prior workload/build differs')
  prior_checks+=audited['cpu_checks']
 runs=[];checks=0;baseline_outputs=None
 for index,run in enumerate(value['runs']):
  native,audited,manifest=audited_pixels_run(ROOT/run['artifacts'],ROOT/run['memory_artifacts'],guard,run['mode'])
  if manifest!=prior['model_manifest_sha256'] or native['binary_sha256']!=value['binary_sha256'] or native['checker_sha256']!=value['checker_sha256'] or native['steps']!=1 or [r['reference_case'] for r in native['requests']]!=[0,0,0,0] or native['requests']!=run['requests'] or native['process_seconds']!=run['process_seconds']:raise RuntimeError('formal raw arm differs')
  outputs=[(r['trace_sha256'],r['png_sha256']) for r in native['requests']]
  if baseline_outputs is None:baseline_outputs=outputs
  if outputs!=baseline_outputs:raise RuntimeError('formal intermediate F32 or PNG bytes differ')
  bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',(ROOT/run['artifacts']/'native.log').read_text(),re.S)
  image_times=[]
  for body,request in zip(bodies,native['requests']):
   image=re.findall(r'SD_PROFILE stage=pipeline part=image_output seconds=([0-9.]+) calls=(\d+)',body)
   if len(image)!=1 or int(image[0][1])!=1 or not 0<float(image[0][0])<request['request_seconds']:raise RuntimeError('one bounded image-output timing required')
   image_times.append(float(image[0][0]))
  checks+=audited['cpu_checks'];audited.update(mode=run['mode'],process_seconds=native['process_seconds'],image_output_seconds=image_times);runs.append(audited)

 metrics={}
 for key,idx in [('process',None)]+[('request'+str(i),i) for i in range(4)]:
  before=sum(r['process_seconds'] if idx is None else r['request_seconds'][idx] for r in runs if r['mode']=='generic')/2
  after=sum(r['process_seconds'] if idx is None else r['request_seconds'][idx] for r in runs if r['mode']=='ve')/2
  metrics[key]=dict(generic_seconds=before,ve_seconds=after,latency_reduction_percent=(1-after/before)*100)
 hot={}
 for mode in ('generic','ve'):
  samples=[seconds for r in runs if r['mode']==mode for seconds in r['request_seconds'][1:]]
  hot[mode]=dict(samples_seconds=samples,mean_seconds=sum(samples)/len(samples))
 hot['latency_reduction_percent']=(1-hot['ve']['mean_seconds']/hot['generic']['mean_seconds'])*100
 if metrics!=value['metrics'] or not value['all_trace_and_png_bytes_identical']:raise RuntimeError('formal metrics or identical-output status differ')
 stage_comparison={}
 for stage in ('clip','unet','vae'):
  for part in ('backend_CPU','backend_BLAS','graph_compute','nlc_thread_prepare','nlc_thread_release'):
   averages={}
   for mode in ('generic','ve'):
    times=[]
    for run in value['runs']:
     if run['mode']==mode:
      for request in run['requests'][1:]:times.append(sum(p['seconds'] for p in request['profile'] if p['stage']==stage and p['part']==part))
    averages[mode]=sum(times)/len(times)
   stage_comparison[stage+'/'+part]=averages
 build_dir=ROOT/'build/sd-baseline-ve/ggml/src/ggml-blas/CMakeFiles/ggml-blas.dir'
 flags_path=build_dir/'flags.make';make_path=build_dir/'build.make'
 flags=re.findall(r'^CXX_FLAGS = (.*)$',flags_path.read_text(),re.M)
 commands=[line for line in make_path.read_text().splitlines() if '$(CXX_FLAGS)' in line and ' -c ' in line and 'sd-ggml-blas.cpp' in line]
 objects=list(build_dir.rglob('sd-ggml-blas.cpp.o'))
 if len(flags)!=1 or len(commands)!=1 or len(objects)!=1:raise RuntimeError('unique actual backend compilation evidence required')
 levels=re.findall(r'(?<!\S)-O[0-3sg](?!\S)',flags[0]+' '+commands[0])
 if not levels:raise RuntimeError('actual backend optimization level required')
 compiler=dict(effective_backend_optimization=levels[-1],backend_flags=flags[0],backend_object_sha256=sha(objects[0]),flags_make_sha256=sha(flags_path),build_make_sha256=sha(make_path))
 image_output={mode:sum(t for r in runs if r['mode']==mode for t in r['image_output_seconds'][1:])/6 for mode in ('generic','ve')}
 png_compiler=png_compiler_evidence();pixels_compiler=pixels_compiler_evidence()
 if pixels_compiler!=prior['actual_pixels_compiler']:raise RuntimeError('actual pixels object/flags changed since model validation')
 if png_compiler!=prior['actual_png_compiler']:raise RuntimeError('actual PNG object/flags changed since model validation')
 files=[folder/'summary.json',folder/'benchmark_sd_resident.py',proof]

 output=dict(status='formal_abba_verified',actual_backend_compiler=compiler,actual_pixels_compiler=pixels_compiler,actual_png_compiler=png_compiler,metrics=metrics,hot_requests=hot,hot_image_output_mean_seconds=image_output,hot_stage_comparison=stage_comparison,runs=runs,binary_sha256=value['binary_sha256'],checker_sha256=value['checker_sha256'],benchmark_sha256=value['benchmark_sha256'],model_manifest_sha256=prior['model_manifest_sha256'],prior_proof=str(proof.relative_to(ROOT)),prior_proof_sha256=sha(proof),prior_cpu_checks=prior_checks,formal_cpu_checks=checks,independent_cpu_checks=prior_checks+checks,all_trace_and_png_bytes_identical=True,publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_model.py','scripts/record_sd_rgb_model.py','scripts/record_sd_png_model.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')},artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},temperature=runs[0]['temperature'],fan_detail=runs[0]['fan_detail'],scope='one VE fixed FP32 SD-Turbo 512x512 one-step prompt, same binary/checker generic/ve/ve/generic, four requests per arm with three hot requests; six hot samples per mode; six-prompt and four-step correctness separate; cold loading cache uncontrolled; hot latency is primary comparison; isolated O2 pixels, fixed RGB resident/PNG O2 with original main O0; default remains original pixel path')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('Formal pixels ABBA audited:',target.relative_to(ROOT));print(metrics)
if __name__=='__main__':main()
