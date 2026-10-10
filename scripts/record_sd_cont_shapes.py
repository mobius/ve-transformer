"""Audit diagnostic CONT geometry/timers and independently recompute model output."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
from record_sd_gelu_result import check_requests
from record_sd_im2col_rows_abba import rows_dispatch
from record_sd_gemm_spatial_model import tile_dispatch
from record_sd_png_model import png_dispatch
from record_sd_pixels_pack_model import recompute_cpu_reference
ROOT=Path(__file__).resolve().parents[1]
FIELDS={'stage','node','threads','type','a','d','an','dn','elements','source_span','destination_span','source_contiguous','destination_contiguous','exact','overlap','seconds'}

def contiguous(ne,nb):
 next_nb=4
 if ne[0]!=1 and nb[0]!=4:return False
 next_nb*=ne[0]
 for i in range(1,4):
  if ne[i]!=1:
   if nb[i]!=next_nb:return False
   next_nb*=ne[i]
 return True

def parse_shapes(body):
 nodes=[]
 for line in body.splitlines():
  if not line.startswith('SD_CONT_SHAPE '):continue
  pairs=[word.split('=',1) for word in line.split()[1:]]
  row=dict(pairs)
  if len(pairs)!=len(FIELDS) or set(row)!=FIELDS or row['stage'] not in ('clip','unet','vae'):raise RuntimeError('complete bounded CONT shape row required')
  for key in ('type','a','d','an','dn'):row[key]=[int(x) for x in row[key].split(',')]
  for key in ('node','threads','elements','source_span','destination_span','source_contiguous','destination_contiguous','exact','overlap'):row[key]=int(row[key])
  row['seconds']=float(row['seconds'])
  if row['type']!=[0,0] or any(len(row[k])!=4 for k in ('a','d','an','dn')) or row['threads']!=8 or row['node']<0 or not math.isfinite(row['seconds']) or row['seconds']<0:raise RuntimeError('expected actual F32 CONT team/geometry/timing differs')
  if any(x<1 for k in ('a','d') for x in row[k]) or any(x<0 or x%4 for k in ('an','dn') for x in row[k]):raise RuntimeError('invalid F32 dimensions/byte strides')
  for ne,nb,span,flag in (('a','an','source_span','source_contiguous'),('d','dn','destination_span','destination_contiguous')):
   if math.prod(row[ne])!=row['elements'] or 4+sum((n-1)*b for n,b in zip(row[ne],row[nb]))!=row[span] or int(contiguous(row[ne],row[nb]))!=row[flag]:raise RuntimeError('independently derived CONT elements/span/contiguity differs')
  if row['destination_contiguous']!=1 or any(row[k] not in (0,1) for k in ('exact','overlap')) or row['exact']>row['overlap']:raise RuntimeError('destination/alias semantics differ')
  nodes.append(row)
 if not nodes or len(nodes)!=body.count('SD_CONT_SHAPE '):raise RuntimeError('complete CONT shape records required')
 return nodes

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run,memory,guard=[x.resolve() for x in (a.native,a.memory,a.guard_log)]
 for path in (run,memory,guard):path.relative_to(ROOT/'build')
 target=a.output.resolve();target.relative_to(ROOT/'docs/results')
 value=json.loads((run/'summary.json').read_text());manifest_path=ROOT/'build/sd-baseline-ve/manifest.json';manifest=json.loads(manifest_path.read_text())
 for name,digest in manifest['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model manifest changed')
 fixed={'mode':'resident','tokenizer_mode':'resident','nlc_threads':'unified','nlc_mode':'openmp','vae_threads':8,'vae_blas_threads':4,'gelu_mode':'ve','binary_scalar_mode':'ve','im2col_mode':'rows_256','vae_spatial_tile':'8192','png_encoder':'ve','rgb_buffer':'resident','pixel_kernel':'ve','operator_profile_enabled':True,'binary_shape_profile_enabled':False,'image_output_profile_enabled':False}
 if any(value.get(k)!=v for k,v in fixed.items()) or value['steps']!=1 or [r['reference_case'] for r in value['requests']]!=[0,0]:raise RuntimeError('fixed CONT diagnostic workload differs')
 checks=check_requests(run,value,manifest['sha256']['build/sd-baseline-ve/bin/sd'],sha(ROOT/'tests/check_sd_resident.py'));rows_dispatch(run,value,'rows_256');tile_dispatch(run,value,'8192');png_dispatch(run,value,'ve')
 raw=(run/'native.log').read_text()
 if any(marker in raw for marker in ('SD_NLC_GEMM ','SD_IM2COL_SHAPE ','SD_IMAGE_OUTPUT_PROFILE ','SD_BINARY_SHAPE ')):raise RuntimeError('unrelated detailed timing probes present')
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',raw,re.S);requests=[]
 if len(bodies)!=2:raise RuntimeError('two ordered request bodies required')
 for request,body in zip(value['requests'],bodies):
  operators=[dict(stage=s,op=o,seconds=float(t),nodes=int(n)) for s,o,t,n in re.findall(r'SD_OP_PROFILE stage=(clip|unet|vae) op=([A-Z_0-9]+) seconds=([0-9.]+) nodes=(\d+)',body)]
  if operators!=request['operator_profile']:raise RuntimeError('raw operator summary differs')
  nodes=parse_shapes(body);stages={}
  for stage in ('clip','unet','vae'):
   subset=[r for r in nodes if r['stage']==stage];op=[r for r in operators if r['stage']==stage and r['op']=='CONT'];total=sum(r['seconds'] for r in subset)
   if len(subset)!=sum(r['nodes'] for r in op) or abs(total-sum(r['seconds'] for r in op))>1e-8:raise RuntimeError('per-node CONT counts/timers differ from independent operator aggregate')
   stages[stage]=dict(nodes=len(subset),seconds=total,logical_copy_bytes=sum(r['elements']*4 for r in subset))
  rgb=re.findall(r'SD_RGB_BUFFER mode=resident allocated=(0|1) reused=(0|1) bytes=786432',body);index=request['request']
  pixels=re.findall(r'SD_PIXEL_KERNEL mode=ve clamp_calls=1 pack_calls=1 spatial=262144 bytes=786432',body)
  if rgb!=[(str(int(index==0)),str(int(index>0)))] or len(pixels)!=1:raise RuntimeError('actual reused RGB/pixel dispatch differs')
  requests.append(dict(request=index,request_seconds=request['request_seconds'],stages=stages,nodes=nodes))
 signature=lambda r:[{k:v for k,v in n.items() if k!='seconds'} for n in r['nodes']]
 if signature(requests[0])!=signature(requests[1]):raise RuntimeError('cold/hot actual CONT geometry differs')
 recomputed,png_count,reference_sha=recompute_cpu_reference(run,value)
 if checks!=10 or len(recomputed)!=10 or png_count!=2:raise RuntimeError('complete independent CPU/PNG audit required')
 sampled=json.loads((memory/'summary.json').read_text());rows=list(csv.DictReader((memory/'memory.csv').open()))
 if not sampled['completed'] or sampled['returncode'] or sampled['final_used_kib']!=131072 or not rows or int(rows[-1]['used_kib'])!=131072 or max(int(r['used_kib']) for r in rows)!=sampled['sampled_highest_used_kib']:raise RuntimeError('completed memory recovery required')
 from record_sd_pixels_pack_abba import guarded_csv
 thermal_paths=guarded_csv(guard)
 old_path=ROOT/'docs/results/20261009T181647Z-sd-turbo-pixels-pack-model.json';old=json.loads(old_path.read_text());prior=next(r for r in old['tests'] if r['name']=='double');base=json.loads((ROOT/prior['artifacts']/'summary.json').read_text())
 if [(r['trace_sha256'],r['png_sha256']) for r in value['requests']]!=[(r['trace_sha256'],r['png_sha256']) for r in base['requests']]:raise RuntimeError('diagnostic trace/PNG differs from accepted model')
 for name,digest in prior['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('accepted prior request evidence changed')
 old_manifest=ROOT/'build/accepted/pixels-pack-model-20261009T181647Z/build/sd-baseline-ve/manifest.json'
 if sha(old_manifest)!=old['model_manifest_sha256']:raise RuntimeError('accepted snapshot manifest differs')
 old_inputs=json.loads(old_manifest.read_text())['sha256'];new_inputs=manifest['sha256'];changed={n for n in old_inputs if old_inputs[n]!=new_inputs.get(n)}
 if set(old_inputs)!=set(new_inputs) or changed!={'scripts/prepare_sd_baseline_overlay.py','build/sd-baseline-overlay/sd-ggml-cpu.c','build/sd-baseline-ve/bin/sd'}:raise RuntimeError('diagnostic build exceeds isolated CPU probe change')
 groups=defaultdict(list)
 for row in requests[1]['nodes']:
  key=json.dumps({k:row[k] for k in ('stage','type','a','d','an','dn','source_contiguous','destination_contiguous','exact','overlap')},sort_keys=True);groups[key].append(row)
 grouped=[]
 for key,nodes in groups.items():grouped.append({**json.loads(key),'nodes':len(nodes),'seconds':sum(n['seconds'] for n in nodes),'logical_copy_bytes':sum(n['elements']*4 for n in nodes)})
 files=[run/'summary.json',run/'native.log',run/'check_sd_resident.py',memory/'summary.json',memory/'memory.csv',guard,manifest_path,old_path,old_manifest]+thermal_paths
 for request in value['requests']:
  output=run/('request'+str(request['request']));files.extend(output/n for n in request['trace_sha256']);files.append(output/'image.png')
 files.extend(ROOT/n for n in prior['sha256'])
 report=dict(status='cont_shape_profile_verified',manifest_sha256=sha(manifest_path),binary_sha256=value['binary_sha256'],checker_sha256=value['checker_sha256'],changed_build_inputs=sorted(changed),requests=requests,hot_shape_groups=sorted(grouped,key=lambda r:r['seconds'],reverse=True),request_seconds=[r['request_seconds'] for r in value['requests']],cpu_recomputed_checks=recomputed,independent_cpu_checks=10,cpu_recomputed_png_checks=2,cpu_reference_sha256=reference_sha,packing_input_domain_verified=True,previous_model_bytes_identical=True,temperature=thermal(guard),fan_detail=fan_detail(guard),sampled_node_peak_gib=sampled['sampled_highest_used_kib']/1048576,final_node_used_mib=128,artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_model.py','scripts/record_sd_pixels_pack_abba.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py','scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_png_model.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py')},scope='single-VE one fixed one-step 512x512 prompt, cold/hot requests with explicit operator/CONT node timing; includes barrier/probe overhead, no optimized data path or speedup claim')
 safe(report);target.write_text(json.dumps(report,indent=2)+'\n');print('CONT shape profile audited:',target.relative_to(ROOT));print('Hot CONT stages:',requests[1]['stages'])
if __name__=='__main__':main()
