"""Audit actual-VE output substages and unchanged prior accepted outputs."""
import argparse
import json
import math
from pathlib import Path
import re
from record_sd_gemm_spatial_model import audited_run
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
ROOT=Path(__file__).resolve().parents[1]
PARTS=['prepare','pixel_clamp','decoded_trace','pixel_pack','png_encode']
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--guard-log',type=Path,required=True);p.add_argument('--build-log',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=[a.run.resolve(),a.memory.resolve(),a.guard_log.resolve()]
 for path in paths:path.relative_to(ROOT/'build')
 value,entry,manifest=audited_run(*paths,'8192')
 if value['steps']!=1 or [r['reference_case'] for r in value['requests']]!=[0,0] or not value.get('image_output_profile_enabled'):raise RuntimeError('matching two-request diagnostic required')
 baseline=a.baseline.resolve();baseline.relative_to(ROOT/'docs/results');old=json.loads(baseline.read_text())
 if old['status']!='formal_abba_verified' or not old['all_trace_and_png_bytes_identical']:raise RuntimeError('accepted formal baseline required')
 root_files=[ROOT/name for name in old['artifact_sha256'] if name.endswith('sd-gemm-spatial-abba/summary.json')]
 if len(root_files)!=1 or sha(root_files[0])!=old['artifact_sha256'][str(root_files[0].relative_to(ROOT))]:raise RuntimeError('frozen baseline summary differs')
 old_run=next(r for r in json.loads(root_files[0].read_text())['runs'] if r['mode']=='8192')
 bodies=re.findall(r'SD_REQUEST_BEGIN index=\d+ resident=\d+\n(.*?)SD_REQUEST_END index=\d+ seconds=[0-9.]+',(paths[0]/'native.log').read_text(),re.S)
 requests=[]
 for index,(body,request) in enumerate(zip(bodies,value['requests'])):
  rows=re.findall(r'SD_IMAGE_OUTPUT_PROFILE part=(\w+) seconds=([0-9.]+) calls=(\d+)',body)
  parsed=[dict(part=r[0],seconds=float(r[1]),calls=int(r[2])) for r in rows]
  total=re.findall(r'SD_PROFILE stage=pipeline part=image_output seconds=([0-9.]+) calls=(\d+)',body)
  if [r['part'] for r in parsed]!=PARTS or any(r['calls']!=1 or not math.isfinite(r['seconds']) or r['seconds']<0 for r in parsed) or parsed!=request['image_output_profile'] or len(rows)!=body.count('SD_IMAGE_OUTPUT_PROFILE ') or len(total)!=1 or int(total[0][1])!=1:raise RuntimeError('raw output substages differ')
  summed=sum(r['seconds'] for r in parsed);whole=float(total[0][0])
  if not 0<summed<=whole+.00001 or not 0<whole<request['request_seconds']:raise RuntimeError('substage timers exceed enclosing request')
  reference=old_run['requests'][index]
  if request['trace_sha256']!=reference['trace_sha256'] or request['png_sha256']!=reference['png_sha256']:raise RuntimeError('diagnostic changed prior F32 or PNG output')
  requests.append(dict(request=index,request_seconds=request['request_seconds'],image_output_seconds=whole,substage_sum_seconds=summed,unattributed_output_seconds=whole-summed,substage=parsed))
 hot=requests[1]
 output=dict(status='image_output_profile_verified',speedup_measured=False,binary_sha256=value['binary_sha256'],checker_sha256=value['checker_sha256'],model_manifest_sha256=manifest,cpu_checks=entry['cpu_checks'],requests=requests,hot_parts_by_seconds=sorted(hot['substage'],key=lambda r:r['seconds'],reverse=True),prior_output_arrays_and_png_byte_identical=True,prior_formal_result=str(baseline.relative_to(ROOT)),prior_formal_result_sha256=sha(baseline),temperature=entry['temperature'],build_temperature=thermal(a.build_log.resolve()),fan_detail=entry['fan_detail'],sampled_node_peak_gib=entry['sampled_node_peak_gib'],final_node_used_mib=128,artifact_sha256=entry['sha256']|{str(baseline.relative_to(ROOT)):sha(baseline),str(a.build_log.resolve().relative_to(ROOT)):sha(a.build_log.resolve())},publisher_sha256=sha(Path(__file__)),audit_dependency_sha256={name:sha(ROOT/name) for name in ('scripts/record_sd_gemm_spatial_model.py','scripts/record_sd_gelu_result.py','scripts/record_sd_im2col_rows_abba.py')},scope='single VE optimized FP32 SD-Turbo 512 one step, same prompt cold/hot, default-disabled substage diagnostics; no speedup claim, image-output substages include trace-file save and PNG encoding; current build CPU checks only')
 safe(output);target=a.output.resolve();target.relative_to(ROOT/'docs/results');target.write_text(json.dumps(output,indent=2)+'\n');print('Image output profile audited:',target.relative_to(ROOT));print(hot)
if __name__=='__main__':main()
