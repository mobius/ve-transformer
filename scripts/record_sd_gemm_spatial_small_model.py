"""Audit actual mixed spatial blocks, retained objects and full CPU/PNG outputs."""
import argparse,csv,json,os,re,subprocess
from collections import Counter
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_model import recompute_cpu_reference
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch
from record_sd_png_model import png_dispatch,png_compiler_evidence
from record_sd_pixels_pack_abba import guarded_csv
from record_sd_group_norm_model import cont_dispatch,compiler_evidence,softmax_dispatch,softmax_compiler_evidence,group_norm_dispatch,group_norm_compiler_evidence
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
BEFORE=ROOT/'build/diagnostics/gemm-small-model-before-20261009T230114Z'
EXPECTED=Counter({('vae',256,262144,2304,8192,32):1,('vae',512,65536,4608,8192,8):1,('vae',256,65536,2304,8192,8):5,('vae',256,65536,4608,8192,8):1,('vae',128,262144,2304,8192,32):1})
MIXED=Counter({('vae',256,262144,2304,8192,32):1,('vae',512,65536,4608,8192,8):1,('vae',256,65536,2304,4096,16):5,('vae',256,65536,4608,4096,16):1,('vae',128,262144,2304,8192,32):1})
def tile_dispatch(folder,value,mode):
 raw=(folder/'native.log').read_text()
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S)
 if len(bodies)!=len(value['requests']) or value.get('vae_spatial_tile')!=mode:raise RuntimeError('tiling mode/request count differs')
 for body,request in zip(bodies,value['requests']):
  rows=re.findall(r'SD_NLC_SPATIAL_TILE stage=(vae) m=(\d+) n=(\d+) k=(\d+) tile=(\d+) calls=(\d+)',body)
  observed=Counter((r[0],)+tuple(map(int,r[1:])) for r in rows)
  parsed=[dict(stage=r[0],m=int(r[1]),n=int(r[2]),k=int(r[3]),tile=int(r[4]),calls=int(r[5])) for r in rows]
  if observed!=(MIXED if mode=='mixed4096' else EXPECTED if mode=='8192' else Counter()) or len(rows)!=body.count('SD_NLC_SPATIAL_TILE ') or parsed!=request.get('vae_spatial_dispatch'):raise RuntimeError('actual tiling shape/call dispatch differs')
def audited_run(run,memory,guard,binary,checker,steps,cases,mode='mixed4096'):
 value=json.loads((run/'summary.json').read_text())
 fixed={'mode':'resident','tokenizer_mode':'resident','nlc_threads':'unified','nlc_mode':'openmp','vae_threads':8,'vae_blas_threads':4,'gelu_mode':'ve','binary_scalar_mode':'ve','im2col_mode':'rows_256','vae_spatial_tile':mode,'png_encoder':'ve','rgb_buffer':'resident','pixel_kernel':'ve'}
 if any(value.get(k)!=v for k,v in fixed.items()) or value['steps']!=steps or [r['reference_case'] for r in value['requests']]!=cases or any(value.get(k) for k in ('operator_profile_enabled','binary_shape_profile_enabled','image_output_profile_enabled')):raise RuntimeError('fixed optimized workload differs')
 checks=check_requests(run,value,binary,checker);rows_dispatch(run,value,'rows_256');tile_dispatch(run,value,mode);png_dispatch(run,value,'ve')
 raw=(run/'native.log').read_text()
 if any(marker in raw for marker in ('SD_NLC_GEMM ','SD_IM2COL_SHAPE ','SD_IMAGE_OUTPUT_PROFILE ')):raise RuntimeError('unexpected detailed timing probes')
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S);image_times=[]
 for index,(body,request) in enumerate(zip(bodies,value['requests'])):
  reused=int(index>0);rgb=re.findall(r'SD_RGB_BUFFER mode=(request|resident) allocated=(0|1) reused=(0|1) bytes=(\d+)',body)
  pixels=re.findall(r'SD_PIXEL_KERNEL mode=(generic|ve) clamp_calls=(\d+) pack_calls=(\d+) spatial=(\d+) bytes=(\d+)',body)
  if rgb!=[('resident',str(1-reused),str(reused),'786432')] or body.count('SD_RGB_BUFFER ')!=1 or request['rgb_buffer_dispatch']!=dict(mode='resident',allocated=1-reused,reused=reused,bytes=786432) or pixels!=[('ve','1','1','262144','786432')] or body.count('SD_PIXEL_KERNEL ')!=1 or request['pixel_kernel_dispatch']!=dict(mode='ve',clamp_calls=1,pack_calls=1,spatial=262144,bytes=786432):raise RuntimeError('actual RGB/pixel dispatch differs')
  images=re.findall(r'SD_PROFILE stage=pipeline part=image_output seconds=([0-9.]+) calls=(\d+)',body)
  if len(images)!=1 or images[0][1]!='1' or not 0<float(images[0][0])<request['request_seconds']:raise RuntimeError('bounded unique output timer required')
  image_times.append(float(images[0][0]))
 recomputed,png_count,reference_sha=recompute_cpu_reference(run,value)
 if len(recomputed)!=checks:raise RuntimeError('CPU audit count differs')
 sampled=json.loads((memory/'summary.json').read_text());rows=list(csv.DictReader((memory/'memory.csv').open()))
 if not sampled['completed'] or sampled['returncode'] or sampled['final_used_kib']!=131072 or not rows or int(rows[-1]['used_kib'])!=131072 or max(int(r['used_kib']) for r in rows)!=sampled['sampled_highest_used_kib']:raise RuntimeError('memory completion/recovery differs')
 files=[run/'summary.json',run/'native.log',run/'check_sd_resident.py',memory/'summary.json',memory/'memory.csv',guard]+guarded_csv(guard)
 for request in value['requests']:
  output=run/('request'+str(request['request']));files.extend(output/name for name in request['trace_sha256']);files.append(output/'image.png')
 entry=dict(packing_input_domain_verified=True,artifacts=str(run.relative_to(ROOT)),memory_artifacts=str(memory.relative_to(ROOT)),request_seconds=[r['request_seconds'] for r in value['requests']],process_seconds=value['process_seconds'],image_output_seconds=image_times,cpu_checks=checks,cpu_recomputed_checks=recomputed,cpu_recomputed_png_checks=png_count,temperature=thermal(guard),fan_detail=fan_detail(guard),sampled_node_peak_gib=sampled['sampled_highest_used_kib']/1048576,sha256={str(p.relative_to(ROOT)):sha(p) for p in files},cpu_reference_sha256=reference_sha)
 return value,entry

def spatial_compiler_evidence():
 from prepare_sd_gemm_spatial_small_model import instrument
 snapshot=json.loads((BEFORE/'archive-sha256.json').read_text())
 accepted_path=ROOT/'docs/results/20261009T222649Z-sd-turbo-group-norm-abba.json';accepted=json.loads(accepted_path.read_text())
 if accepted['status']!='formal_group_norm_abba_verified' or not accepted.get('cpu_validation_temperature') or sha(BEFORE/'build/sd-baseline-ve/manifest.json')!=accepted['model_manifest_sha256'] or sha(BEFORE/'build/sd-baseline-ve/bin/sd')!=accepted['binary_sha256'] or sha(BEFORE/'tests/check_sd_resident.py')!=accepted['checker_sha256']:raise RuntimeError('frozen before-build model must match accepted formal proof')
 old_inputs=json.loads((BEFORE/'build/sd-baseline-ve/manifest.json').read_text())['sha256']
 for name,digest in old_inputs.items():
  if sha(BEFORE/name)!=digest:raise RuntimeError('accepted before-build model input differs')
 for name,digest in snapshot.items():
  if sha(BEFORE/name)!=digest:raise RuntimeError('before-build snapshot changed')
 source=ROOT/'build/sd-baseline-overlay/sd-ggml-blas.cpp'
 if instrument((BEFORE/'build/sd-baseline-overlay/sd-ggml-blas.cpp').read_text())!=source.read_text():raise RuntimeError('BLAS delta exceeds selective block change')
 folder=ROOT/'build/sd-baseline-ve/ggml/src/ggml-blas/CMakeFiles/ggml-blas.dir'
 objects=list(folder.rglob('sd-ggml-blas.cpp.o'));flags=folder/'flags.make';make=folder/'build.make'
 commands=[r for r in make.read_text().splitlines() if '$(CXX_FLAGS)' in r and ' -c ' in r and 'sd-ggml-blas.cpp' in r]
 base_flags=re.findall(r'^CXX_FLAGS = (.*)$',flags.read_text(),re.M)
 if len(objects)!=1 or len(commands)!=1 or len(base_flags)!=1:raise RuntimeError('unique actual BLAS build required')
 effective=base_flags[0]+' '+commands[0]
 if re.findall(r'(?<!\S)-O[0-3sg](?!\S)',effective)[-1:]!=['-O1'] or '-fno-fast-math' not in effective or '-fopenmp' not in effective:raise RuntimeError('original BLAS compiler mode required')
 obj=objects[0];archive=ROOT/'build/sd-baseline-ve/ggml/src/ggml-blas/libggml-blas.a'
 if obj.stat().st_mtime_ns<source.stat().st_mtime_ns or subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()!=['sd-ggml-blas.cpp.o']:raise RuntimeError('fresh unique archive member required')
 import hashlib
 if hashlib.sha256(subprocess.check_output(['ar','p',str(archive),'sd-ggml-blas.cpp.o'])).hexdigest()!=sha(obj):raise RuntimeError('actual archive member differs')
 link=ROOT/'build/sd-baseline-ve/examples/cli/CMakeFiles/sd.dir/link.txt'
 if 'libggml-blas.a' not in link.read_text():raise RuntimeError('BLAS archive absent from actual link')
 def symbols(path):
  rows=[]
  for line in subprocess.check_output(['readelf','-Ws',str(path)],text=True).splitlines():
   parts=line.split()
   if len(parts)>=8 and parts[3]=='FUNC' and 'ggml_backend_blas_graph_compute' in parts[7]:rows.append((parts[7],int(parts[2])))
  return rows
 sizes=symbols(obj)
 if len(sizes)!=1 or sizes!=symbols(ROOT/'build/sd-baseline-ve/bin/sd'):raise RuntimeError('actual linked BLAS graph function differs')
 changed=[];unchanged=0
 for name,digest in snapshot.items():
  if Path(name).suffix not in ('.o','.a'):continue
  if sha(ROOT/name)!=digest:
   if name not in (str(obj.relative_to(ROOT)),str(archive.relative_to(ROOT))):raise RuntimeError('unrelated model object changed')
   changed.append(Path(name).name)
  else:unchanged+=1
 if sorted(changed)!=['libggml-blas.a','sd-ggml-blas.cpp.o']:raise RuntimeError('only intended BLAS object/archive delta required')
 old=json.loads((BEFORE/'build/sd-baseline-ve/manifest.json').read_text());now=json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
 added=set(now['sha256'])-set(old['sha256']);removed=set(old['sha256'])-set(now['sha256']);delta={name for name in now['sha256'] if name in old['sha256'] and now['sha256'][name]!=old['sha256'][name]}
 if removed or added!={'scripts/prepare_sd_gemm_spatial_small_model.py'} or delta!={'scripts/prepare_sd_baseline_overlay.py','scripts/record_sd_baseline_build.py','build/sd-baseline-overlay/sd-ggml-blas.cpp','build/sd-baseline-ve/bin/sd'}:raise RuntimeError('model manifest delta exceeds small spatial integration')
 independent=ROOT/'docs/results/20261009T225206Z-sd-turbo-gemm-spatial-small.json';v=json.loads(independent.read_text())
 if v['status']!='current_baseline_microbenchmark_verified' or not v.get('cpu_validation_temperature') or v['full_blas_comparisons']!=288 or v['independent_fp64_samples']!=5760:raise RuntimeError('finalized small block probe required')
 for shape in (3,4):
  row=next(r for r in v['comparisons'] if r['shape']==shape and r['tile']==4096)
  if not row['both_pairs_positive'] or row['latency_reduction_percent']<=0 or row['baseline_tile']!=8192:raise RuntimeError('selected candidate evidence differs')
 libraries={n:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/n) for n in ('libcblas.so','libblas_openmp.so')}
 if libraries!=v['nlc_library_sha256']:raise RuntimeError('NLC differs from independent probe')
 return dict(accepted_before_proof_sha256=sha(accepted_path),accepted_before_manifest_sha256=accepted['model_manifest_sha256'],object_sha256=sha(obj),archive_sha256=sha(archive),source_sha256=sha(source),flags_make_sha256=sha(flags),build_make_sha256=sha(make),link_sha256=sha(link),linked_function_size=sizes[0][1],only_intended_source_delta=True,changed_objects=sorted(changed),unchanged_objects=unchanged,added_model_inputs=sorted(added),changed_model_inputs=sorted(delta),before_snapshot_index_sha256=sha(BEFORE/'archive-sha256.json'),independent_proof_sha256=sha(independent),nlc_library_sha256=libraries)

def main():
 if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':raise RuntimeError('temperature supervision required')
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--name',choices=('double','six','four'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();proof.relative_to(ROOT/'docs/results');paths=[x.resolve() for x in (a.run,a.memory,a.guard_log)]
 for path in paths:path.relative_to(ROOT/'build')
 manifest_path=ROOT/'build/sd-baseline-ve/manifest.json';manifest=json.loads(manifest_path.read_text())
 for name,digest in manifest['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('current model input changed')
 binary=sha(ROOT/'build/sd-baseline-ve/bin/sd');checker=sha(ROOT/'tests/check_sd_resident.py');steps,cases={'double':(1,[0,0]),'six':(1,list(range(6))),'four':(4,[0,0])}[a.name]
 result,entry=audited_run(*paths,binary,checker,steps,cases,'mixed4096')
 entry.update(name=a.name,cont_dispatch=cont_dispatch(paths[0],result,'extended'),softmax_scale_dispatch=softmax_dispatch(paths[0],result),group_norm_dispatch=group_norm_dispatch(paths[0],result))
 compilers=dict(actual_spatial_compiler=spatial_compiler_evidence(),actual_cont_compiler=compiler_evidence(),actual_png_compiler=png_compiler_evidence(),actual_softmax_scale_compiler=softmax_compiler_evidence(),actual_group_norm_compiler=group_norm_compiler_evidence())
 value=json.loads(proof.read_text()) if proof.exists() else dict(status='model_validation_in_progress',tests=[],model_manifest_sha256=sha(manifest_path),binary_sha256=binary,checker_sha256=checker)
 if value['model_manifest_sha256']!=sha(manifest_path) or value['binary_sha256']!=binary or value['checker_sha256']!=checker or any(value.get(k,v)!=v for k,v in compilers.items()):raise RuntimeError('model/checker/actual objects changed')
 existing=[e for e in value['tests'] if e['name']!=a.name]
 for e in existing:
  for group in ('sha256','cpu_reference_sha256'):
   for name,digest in e[group].items():
    if sha(ROOT/name)!=digest:raise RuntimeError('earlier evidence changed')
 value.update(compilers);value['tests']=sorted(existing+[entry],key=lambda e:('double','six','four').index(e['name']));value['completed_tests']=[e['name'] for e in value['tests']];value['all_candidate_tests_completed']=value['completed_tests']==['double','six','four'];value['status']='model_validation_verified' if value['all_candidate_tests_completed'] else 'model_validation_in_progress';value['independent_cpu_checks']=sum(e['cpu_checks'] for e in value['tests']);value['cpu_recomputed_checks']=sum(len(e['cpu_recomputed_checks']) for e in value['tests']);value['cpu_recomputed_png_checks']=sum(e['cpu_recomputed_png_checks'] for e in value['tests']);value['full_graph_speedup_measured']=False
 value['publisher_sha256']=sha(Path(__file__));value['audit_dependency_sha256']={name:sha(ROOT/name) for name in ('scripts/record_sd_group_norm_model.py','scripts/record_sd_pixels_pack_model.py','scripts/record_sd_pixels_pack_abba.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_png_model.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/prepare_sd_group_norm_model.py','scripts/prepare_sd_gemm_spatial_small_model.py','scripts/benchmark_sd_runtime.py')}
 value['configuration']=dict(vae_spatial_tile='mixed4096',generic_threads=8,vae_blas_threads=4,im2col_mode='rows_256',group_norm='selective',softmax_scale='selective',cont_kernel='extended',pixel_kernel='ve',rgb_buffer='resident',png_encoder='ve',gelu='ve',binary_scalar='ve',weights='resident',tokenizer='resident');value['scope']='actual model two selected 4096 spatial blocks, other three 8192 blocks and original fallback retained; 9 nodes/168 calls per request; unchanged other objects; default off; full graph incremental speedup unmeasured'
 safe(value);proof.write_text(json.dumps(value,indent=2)+'\n');print('Mixed spatial model tests audited:',value['completed_tests'],'CPU checks:',value['independent_cpu_checks'])
if __name__=='__main__':main()
