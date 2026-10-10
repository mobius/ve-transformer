"""Clone the accepted actual backend, changing only Softmax row scaling."""
import json,shlex,shutil
from pathlib import Path
from benchmark_sd_runtime import sha
ROOT=Path(__file__).resolve().parents[1]
def main():
 folder=ROOT/'build/sd-softmax-graph-probe';folder.mkdir(exist_ok=True)
 archive=ROOT/'build/accepted/softmax-shapes-20261009T201945Z';index=json.loads((archive/'archive-sha256.json').read_text());manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if sha(manifest)!=index['build/sd-baseline-ve/manifest.json']:raise RuntimeError('accepted manifest changed')
 for n,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('model input changed')
 names=['build/sd-baseline-ve/ggml/src/libggml.a','build/sd-baseline-ve/ggml/src/libggml-base.a','build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a']
 for n in names:
  if sha(ROOT/n)!=index[n]:raise RuntimeError('actual graph library changed')
 source=ROOT/'build/sd-baseline-overlay/sd-ggml-cpu.c';s=source.read_text();start=s.index('static void ggml_compute_forward_soft_max_f32(');end=s.index('static void ggml_compute_forward_soft_max(',start);old=s[start:end];new=old
 needle='    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);'
 assert new.count(needle)==1;new=new.replace(needle,needle+'\n    const char * scale_flag=getenv("SD_VE_SOFTMAX_SCALE");\n    const bool scale_candidate=scale_flag && strcmp(scale_flag,"1")==0;')
 needle='        ggml_vec_cpy_f32  (nc, wp, sp);\n        ggml_vec_scale_f32(nc, wp, scale);';assert new.count(needle)==1;new=new.replace(needle,'        if (scale_candidate) sd_ve_softmax_copy_scale_f32(nc, wp, sp, scale);\n        else {\n'+needle+'\n        }')
 needle='        ggml_vec_scale_f32(nc, dp, sum);';assert new.count(needle)==1;new=new.replace(needle,'        if (scale_candidate) sd_ve_softmax_scale_f32(nc, dp, (float) sum);\n        else '+needle.strip())
 s=s[:start]+new+s[end:];s='extern void sd_ve_softmax_scale_f32(int,float*,float);\nextern void sd_ve_softmax_copy_scale_f32(int,float*,const float*,float);\n'+s;(folder/'sd-ggml-cpu.c').write_text(s)
 shutil.copy2(ROOT/names[2],folder/'libggml-cpu.a')
 flags=ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir/flags.make';lines=flags.read_text().splitlines();args=[]
 for name in ('C_DEFINES','C_INCLUDES','C_FLAGS'):
  args+=shlex.split(next(l.split(' = ',1)[1] for l in lines if l.startswith(name+' = ')))
 (folder/'compile-args.json').write_text(json.dumps(args))
 proof=dict(manifest_sha256=sha(manifest),source_sha256=sha(source),libraries_sha256={n:sha(ROOT/n) for n in names},cloned_source_sha256=sha(folder/'sd-ggml-cpu.c'),changes='only row copy/scale and normalization; same actual graph API, masks/max/exp/sequential sum unchanged; env switch per node')
 shapes=json.loads((ROOT/'docs/results/20261009T201945Z-sd-turbo-softmax-shapes.json').read_text())['shape_requests'][1]['shapes'];unique=[]
 for r in shapes:
  if r['ne'] not in unique:unique.append(r['ne'])
 if len(unique)!=10:raise RuntimeError('ten actual geometries required')
 (folder/'shapes.tsv').write_text(''.join('\t'.join(map(str,[i]+ne))+'\n' for i,ne in enumerate(unique)))
 proof['shape_proof_sha256']=sha(ROOT/'docs/results/20261009T201945Z-sd-turbo-softmax-shapes.json');proof['shapes']=unique
 (folder/'baseline.json').write_text(json.dumps(proof,indent=2)+'\n');print('Accepted actual ggml backend clone prepared')
if __name__=='__main__':main()
