"""Prepare a standalone actual GroupNorm backend clone for final-row scaling.

Preparation only: this helper neither compiles nor runs hardware. Actual model
shape fixtures must be captured and bound separately before benchmarking.
"""
import argparse,json,shlex,shutil
from pathlib import Path
from benchmark_sd_runtime import sha
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=Path('build/sd-group-norm-scale-probe'));a=p.parse_args();folder=a.folder.resolve();folder.relative_to(ROOT/'build');folder.mkdir(exist_ok=True,parents=True)
 accepted=ROOT/'build/accepted/group-norm-shapes-20261009T213215Z';index=json.loads((accepted/'archive-sha256.json').read_text());manifest=ROOT/'build/sd-baseline-ve/manifest.json';source=ROOT/'build/sd-baseline-overlay/sd-ggml-cpu.c'
 for path in (manifest,source):
  if sha(path)!=index[str(path.relative_to(ROOT))]:raise RuntimeError('accepted actual backend changed')
 for n,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('model input changed')
 text=source.read_text();start=text.index('static void ggml_compute_forward_group_norm_f32(');end=text.index('static void ggml_compute_forward_group_norm(',start);body=text[start:end]
 needle='    int n_groups = dst->op_params[0];'
 if body.count(needle)!=1:raise RuntimeError('unique group count declaration required')
 body=body.replace(needle,needle+'\n    const char * scale_flag=getenv("SD_GROUP_NORM_SCALE");\n    const bool scale_candidate=scale_flag && strcmp(scale_flag,"1")==0;')
 needle="""            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }"""
 if body.count(needle)!=1:raise RuntimeError('unique final GroupNorm scaling block required')
 replacement="""            if (scale_candidate && ggml_is_contiguous(dst) && ne00>0 && ne01>0 && step>0 &&
                    ne00<=2147483647LL && ne01<=2147483647LL/ne00 && step<=2147483647LL/(ne00*ne01)) {
                float * group_y=(float *)((char *)dst->data + start*nb2 + i03*nb3);
                sd_ve_softmax_scale_f32((int)(ne00*ne01*step),group_y,scale);
            } else {
"""+needle+"""
            }"""
 body=body.replace(needle,replacement)
 text=text[:start]+body+text[end:];(folder/'sd-ggml-cpu.c').write_text(text)
 libraries={}
 for n in ('build/sd-baseline-ve/ggml/src/libggml.a','build/sd-baseline-ve/ggml/src/libggml-base.a','build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a'):
  if sha(ROOT/n)!=index[n]:raise RuntimeError('accepted actual graph archive changed')
  libraries[n]=sha(ROOT/n)
 shutil.copy2(ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/libggml-cpu.a',folder/'libggml-cpu.a')
 flags=ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir/flags.make';args=[]
 for name in ('C_DEFINES','C_INCLUDES','C_FLAGS'):args+=shlex.split(next(line.split(' = ',1)[1] for line in flags.read_text().splitlines() if line.startswith(name+' = ')))
 proof_path=ROOT/'docs/results/20261009T213215Z-sd-turbo-group-norm-shapes.json';proof=json.loads(proof_path.read_text())
 if proof['status']!='group_norm_shape_profile_verified' or not proof['previous_model_bytes_identical'] or proof['manifest_sha256']!=sha(manifest):raise RuntimeError('accepted actual GroupNorm shapes required')
 cases=[]
 for r in proof['shape_requests'][1]['shapes']:
  key=(r['stage'],tuple(r['ne']),r['groups'],r['epsilon'])
  if any((c['stage'],tuple(c['ne']),c['groups'],c['epsilon'])==key for c in cases):continue
  ne=r['ne'];nb=[4,4*ne[0],4*ne[0]*ne[1],4*ne[0]*ne[1]*ne[2]]
  if r['src_nb']!=nb or r['dst_nb']!=nb or ne[3]!=1 or ne[2]%r['groups']:raise RuntimeError('observed contiguous divisible GroupNorm geometry required')
  cases.append(dict(id=len(cases),stage=r['stage'],ne=ne,groups=r['groups'],epsilon=r['epsilon']))
 if len(cases)!=24:raise RuntimeError('24 actual combinations required')
 (folder/'shapes.tsv').write_text(''.join('\t'.join(map(str,[c['id']]+c['ne']+[c['groups'],format(c['epsilon'],'.9g')]))+'\n' for c in cases))
 (folder/'fixtures.json').write_text(json.dumps(dict(shape_proof_sha256=sha(proof_path),cases=cases),indent=2)+'\n')
 (folder/'compile-args.json').write_text(json.dumps(args)+'\n');(folder/'baseline.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),source_sha256=sha(source),cloned_source_sha256=sha(folder/'sd-ggml-cpu.c'),libraries_sha256=libraries,accepted_index_sha256=sha(accepted/'archive-sha256.json'),scope='only final group scaling changed; mean/variance sequential FP64 accumulation and FP32 centering/squaring unchanged; 24 real shape/epsilon combinations bound; hardware validation pending'),indent=2)+'\n');print('Standalone GroupNorm scaling source prepared:',folder.relative_to(ROOT))
if __name__=='__main__':main()
