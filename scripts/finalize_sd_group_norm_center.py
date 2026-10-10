"""Bind CPU completion, the isolated source delta and archive math members."""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from benchmark_sd_runtime import sha
from record_sd_pixels_pack_abba import guarded_csv
from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    result.relative_to(ROOT / 'docs/results')
    log = args.log.resolve()
    log.relative_to(ROOT / 'build')
    value = json.loads(result.read_text())
    if ('GroupNorm center CPU audit passed: ' + str(result.relative_to(ROOT)) not in log.read_text()
            or value['independent_cpu_cases'] != 24 or value['ve_bitwise_checks'] != 288
            or value['timed_graph_calls'] != 9216 or value['timed_batches'] != 576
            or value['actual_parallel_team_records'] != 456
            or value['candidate_output_captures'] != 24):
        raise RuntimeError('complete graph, captures and CPU audit required')
    for name, digest in value['artifact_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('bound evidence changed')
    if sha(ROOT / 'scripts/record_sd_group_norm_center.py') != value['publisher_sha256']:
        raise RuntimeError('CPU auditor changed')
    probe = ROOT / 'build/sd-group-norm-center-probe'
    binding = json.loads((probe / 'baseline.json').read_text())
    accepted = ROOT / 'build/accepted/group-norm-batched-20261009T215139Z'
    if sha(accepted / 'archive-sha256.json') != binding['accepted_probe_index_sha256']:
        raise RuntimeError('accepted probe index changed')
    previous = accepted / 'build/sd-group-norm-batched-probe/sd-ggml-cpu.c'
    if sha(previous) != binding['baseline_probe_source_sha256']:
        raise RuntimeError('accepted baseline source changed')
    before = previous.read_text()
    after = (probe / 'sd-ggml-cpu.c').read_text()
    declaration = ('void sd_ve_group_norm_center_square_f32(int n,float *dst,'
                   'float *squares,const float *src,float mean);\n\n')
    flags = ('    const char * center_flag=getenv("SD_GROUP_NORM_CENTER_SQUARE");\n'
             '    const bool center_candidate=center_flag && strcmp(center_flag,"1")==0;\n'
             '    float group_squares[2048];\n')
    if after.count(declaration) != 1 or after.count(flags) != 1:
        raise RuntimeError('unique center declaration and bounded scratch required')
    restored = after.replace(declaration, '').replace(flags, '')
    function = 'static void ggml_compute_forward_group_norm_f32('
    old_start = before.index('            ggml_float sum2 = 0.0;', before.index(function))
    old_end = before.index('            const float variance =', old_start)
    start = restored.index('            ggml_float sum2 = 0.0;', restored.index(function))
    end = restored.index('            const float variance =', start)
    block = restored[start:end]
    original_block = before[old_start:old_end]
    fallback = original_block[len('            ggml_float sum2 = 0.0;\n'):]
    if block.count(fallback) != 1 or not block.endswith(fallback + '            }\n'):
        raise RuntimeError('literal original variance fallback required')
    for fragment in ('offset+=2048', '2048%ne00==0', 'ggml_is_contiguous(src0)',
                     'ggml_is_contiguous(dst)', 'ggml_nbytes(src0)', 'ggml_nbytes(dst)',
                     'row+=(int)ne00', 'sumr += (ggml_float)group_squares[row+col];',
                     'sum2 += sumr;'):
        if fragment not in block:
            raise RuntimeError('bounded row-order candidate invariant missing')
    restored = restored[:start] + original_block + restored[end:]
    if restored != before:
        raise RuntimeError('source changed outside isolated center/square block')
    original = accepted / 'build/sd-group-norm-batched-probe/libggml-cpu.a'
    if sha(original) != binding['baseline_probe_library_sha256']:
        raise RuntimeError('accepted baseline archive changed')
    clone = probe / 'libggml-cpu.a'
    nar = '/opt/nec/ve/bin/nar'
    members = subprocess.check_output([nar, 't', str(original)], text=True).splitlines()
    actual = subprocess.check_output([nar, 't', str(clone)], text=True).splitlines()
    addition = 've_sd_turbo_group_norm.c.o'
    if actual != members + [addition] or len(actual) != len(set(actual)):
        raise RuntimeError('only the new center/square member may be added')
    hashes = {}
    for member in actual:
        candidate = hashlib.sha256(subprocess.check_output([nar, 'p', str(clone), member])).hexdigest()
        baseline = (hashlib.sha256(subprocess.check_output([nar, 'p', str(original), member])).hexdigest()
                    if member in members else None)
        if member in ('sd-ggml-cpu.c.o', addition):
            if candidate != sha(probe / member):
                raise RuntimeError('candidate archive object differs')
        elif candidate != baseline:
            raise RuntimeError('existing mathematical object changed')
        hashes[member] = dict(original=baseline, candidate=candidate)
    make_flags = ROOT / 'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir/flags.make'
    compile_args = []
    for name in ('C_DEFINES', 'C_INCLUDES', 'C_FLAGS'):
        compile_args += shlex.split(next(line.split(' = ', 1)[1]
            for line in make_flags.read_text().splitlines() if line.startswith(name + ' = ')))
    if (compile_args != json.loads((probe / 'compile-args.json').read_text())
            or [x for x in compile_args if re.fullmatch(r'-O[0-3sg]', x)][-1:] != ['-O1']
            or '-fno-fast-math' not in compile_args):
        raise RuntimeError('accepted strict CPU compile flags changed')
    value.update(center_only_delta_verified=True, archive_member_sha256=hashes,
                 cpu_validation_temperature=thermal(log), cpu_validation_fan_detail=fan_detail(log),
                 raw_candidate_label_capture='explicit candidate graph invocation after timed ABBA; full output/source comparison before saving',
                 finalizer_sha256=sha(Path(__file__)))
    files = [log, probe / 'compile-args.json', make_flags, previous, original,
             accepted / 'archive-sha256.json'] + guarded_csv(log)
    value['artifact_sha256'].update({str(f.relative_to(ROOT)): sha(f) for f in files})
    value['audit_dependency_sha256'] = {name: sha(ROOT / name) for name in (
        'scripts/record_sd_pixels_pack_abba.py', 'scripts/record_qwen36_mtp.py',
        'scripts/record_sd_im2col_result.py', 'scripts/benchmark_sd_runtime.py')}
    safe(value)
    result.write_text(json.dumps(value, indent=2) + '\n')
    print('GroupNorm center/square independent CPU audit finalized')


if __name__ == '__main__':
    main()
