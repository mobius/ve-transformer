"""Bind CPU completion and actual archive/compile provenance."""
import argparse,json,re,subprocess,hashlib,shlex
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');v=json.loads(result.read_text())
 if 'GroupNorm scale CPU audit passed: '+str(result.relative_to(ROOT)) not in log.read_text() or v['independent_cpu_cases']!=24 or v['ve_bitwise_checks']!=288 or v['timed_graphs']!=9216 or v['actual_parallel_team_records']!=432 or v['calls_per_batch']!=16 or v['timed_graph_calls']!=9216:raise RuntimeError('complete actual graph and CPU audit required')
 for n,h in v['artifact_sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('evidence changed')
 if sha(ROOT/'scripts/record_sd_group_norm_batched.py')!=v['publisher_sha256']:raise RuntimeError('CPU auditor changed')
 previous=ROOT/'build/sd-group-norm-scale-probe/sd-ggml-cpu.c';current=ROOT/'build/sd-group-norm-batched-probe/sd-ggml-cpu.c'
 marker='                fprintf(stderr,"SD_GGML_THREADS stage=%s actual=%d maximum=%d'+chr(92)+'n",'
 accepted=ROOT/'docs/results/20261009T213934Z-sd-turbo-group-norm-scale.json';prior=json.loads(accepted.read_text())
 if sha(previous)!=prior['artifact_sha256'][str(previous.relative_to(ROOT))]:raise RuntimeError('previous validated clone changed')
 v['previous_validated_probe_sha256']=sha(accepted)
 original_text=previous.read_text()
 if original_text.count(marker)!=1:raise RuntimeError('unique diagnostic required')
 expected=original_text.replace(marker,'                const char * quiet=getenv("SD_GGML_THREADS_QUIET");'+chr(10)+'                if (!(quiet && strcmp(quiet,"1")==0))'+chr(10)+marker)
 if current.read_text()!=expected:raise RuntimeError('candidate math source changed beyond diagnostic conditional')
 v['diagnostic_only_delta_verified']=True
 v['raw_candidate_label_capture']='saved after ABBA final baseline arm; candidate outputs checked against baseline with full memcmp after every candidate batch; CPU reference validates saved baseline and candidate via this equality, not a separately captured candidate invocation'

 v['artifact_sha256'][str(accepted.relative_to(ROOT))]=sha(accepted)
 v['artifact_sha256'][str(previous.relative_to(ROOT))]=sha(previous)
 probe=ROOT/'build/sd-group-norm-batched-probe';binding=json.loads((probe/'baseline.json').read_text());original=ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a';clone=probe/'libggml-cpu.a';nar='/opt/nec/ve/bin/nar'
 members=subprocess.check_output([nar,'t',str(original)],text=True).splitlines()
 if members!=subprocess.check_output([nar,'t',str(clone)],text=True).splitlines() or len(set(members))!=len(members):raise RuntimeError('actual archive member set differs')
 hashes={}
 for member in members:
  before=hashlib.sha256(subprocess.check_output([nar,'p',str(original),member])).hexdigest();after=hashlib.sha256(subprocess.check_output([nar,'p',str(clone),member])).hexdigest()
  if member=='sd-ggml-cpu.c.o':
   if after!=sha(probe/'sd-ggml-cpu.c.o'):raise RuntimeError('cloned CPU object differs')
  elif before!=after:raise RuntimeError('archive math member changed')
  hashes[member]=dict(original=before,candidate=after)
 flags=ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir/flags.make';args=[]
 for name in ('C_DEFINES','C_INCLUDES','C_FLAGS'):args+=shlex.split(next(line.split(' = ',1)[1] for line in flags.read_text().splitlines() if line.startswith(name+' = ')))
 if args!=json.loads((probe/'compile-args.json').read_text()) or [x for x in args if re.fullmatch(r'-O[0-3sg]',x)][-1:]!=['-O1'] or '-fno-fast-math' not in args:raise RuntimeError('actual accepted CPU flags changed')
 v['archive_member_sha256']=hashes;v['cpu_validation_temperature']=thermal(log);v['cpu_validation_fan_detail']=fan_detail(log);v['artifact_sha256'].update({str(f.relative_to(ROOT)):sha(f) for f in [log,probe/'compile-args.json',flags]+guarded_csv(log)});v['audit_dependency_sha256']={n:sha(ROOT/n) for n in ('scripts/record_sd_pixels_pack_abba.py','scripts/record_qwen36_mtp.py','scripts/record_sd_im2col_result.py','scripts/benchmark_sd_runtime.py')};v['finalizer_sha256']=sha(Path(__file__));safe(v);result.write_text(json.dumps(v,indent=2)+'\n');print('GroupNorm scale independent CPU audit finalized')
if __name__=='__main__':main()
