"""Clone the accepted quiet graph probe and add bounded center/square tiles."""
import json
import shutil
from pathlib import Path
from benchmark_sd_runtime import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    accepted = ROOT / 'build/accepted/group-norm-batched-20261009T215139Z'
    index = json.loads((accepted / 'archive-sha256.json').read_text())
    prior = 'build/sd-group-norm-batched-probe'
    folder = ROOT / 'build/sd-group-norm-center-probe'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('baseline.json', 'fixtures.json', 'shapes.tsv', 'compile-args.json',
                 'libggml-cpu.a', 'sd-ggml-cpu.c'):
        source = accepted / prior / name
        if sha(source) != index[prior + '/' + name]:
            raise RuntimeError('accepted probe artifact changed')
        shutil.copy2(source, folder / name)
    binding = json.loads((folder / 'baseline.json').read_text())
    manifest = ROOT / 'build/sd-baseline-ve/manifest.json'
    if sha(manifest) != binding['manifest_sha256']:
        raise RuntimeError('accepted model manifest changed')
    for name, digest in json.loads(manifest.read_text())['sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('accepted model input changed')
    for name, digest in binding['libraries_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('accepted model library changed')
    proof = ROOT / 'docs/results/20261009T215139Z-sd-turbo-group-norm-batched.json'
    if sha(proof) != index[str(proof.relative_to(ROOT))]:
        raise RuntimeError('accepted graph proof changed')
    previous_source = folder / 'sd-ggml-cpu.c'
    text = previous_source.read_text()
    binding['baseline_probe_source_sha256'] = sha(previous_source)
    binding['baseline_probe_library_sha256'] = sha(folder / 'libggml-cpu.a')
    function = 'static void ggml_compute_forward_group_norm_f32('
    declaration = ('void sd_ve_group_norm_center_square_f32(int n,float *dst,'
                   'float *squares,const float *src,float mean);\n\n')
    if text.count(function) != 1:
        raise RuntimeError('unique GroupNorm implementation required')
    text = text.replace(function, declaration + function)
    start = text.index(function)
    end = text.index('static void ggml_compute_forward_group_norm(', start)
    body = text[start:end]
    needle = '    int n_groups = dst->op_params[0];'
    if body.count(needle) != 1:
        raise RuntimeError('unique group declaration required')
    body = body.replace(needle, needle + '\n' +
        '    const char * center_flag=getenv("SD_GROUP_NORM_CENTER_SQUARE");\n'
        '    const bool center_candidate=center_flag && strcmp(center_flag,"1")==0;\n'
        '    float group_squares[2048];')
    begin = body.index('            ggml_float sum2 = 0.0;')
    finish = body.index('            const float variance =', begin)
    original = body[begin:finish]
    # Existing scalar loop is preserved literally as the fallback.
    fallback = original[len('            ggml_float sum2 = 0.0;\n'):]
    replacement = '''            ggml_float sum2 = 0.0;
            if (center_candidate && ggml_is_contiguous(src0) && ggml_is_contiguous(dst) &&
                    ((uintptr_t)dst->data>(uintptr_t)src0->data ?
                        (uintptr_t)dst->data-(uintptr_t)src0->data>=ggml_nbytes(src0) :
                        (uintptr_t)src0->data-(uintptr_t)dst->data>=ggml_nbytes(dst)) && ne00>0 && ne00<=2048 && 2048%ne00==0 &&
                    ne01>0 && step>0 && ne01<=2147483647LL/ne00 &&
                    step<=2147483647LL/(ne00*ne01)) {
                const int64_t group_count=ne00*ne01*step;
                const float * group_x=(const float *)((const char *)src0->data + start*nb02 + i03*nb03);
                float * group_y=(float *)((char *)dst->data + start*nb2 + i03*nb3);
                for (int64_t offset=0; offset<group_count; offset+=2048) {
                    const int count=(int)(group_count-offset<2048 ? group_count-offset : 2048);
                    sd_ve_group_norm_center_square_f32(count,group_y+offset,group_squares,group_x+offset,mean);
                    for (int row=0; row<count; row+=(int)ne00) {
                        ggml_float sumr=0.0;
                        for (int col=0; col<(int)ne00; ++col) {
                            sumr += (ggml_float)group_squares[row+col];
                        }
                        sum2 += sumr;
                    }
                }
            } else {
''' + fallback + '            }\n'
    body = body[:begin] + replacement + body[finish:]
    text = text[:start] + body + text[end:]
    previous_source.write_text(text)
    binding.update(cloned_source_sha256=sha(previous_source),
                   accepted_probe_index_sha256=sha(accepted / 'archive-sha256.json'),
                   accepted_probe_proof_sha256=sha(proof),
                   scope='center/square O2 tiles only; mean and row/outer FP64 variance accumulation order unchanged; group scaling enabled for both modes; actual hardware validation pending')
    (folder / 'baseline.json').write_text(json.dumps(binding, indent=2) + '\n')
    print('Standalone GroupNorm center/square source prepared:', folder.relative_to(ROOT))


if __name__ == '__main__':
    main()
