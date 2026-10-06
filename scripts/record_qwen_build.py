"""Record relative artifact names and hashes; no host identity or environment dump."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(relative):
    path = ROOT / relative
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ve-flags', required=True)
    args = parser.parse_args()
    artifacts = [
        'src/qwen_infer.cpp', 'src/qwen_nlc_hook.cpp', 'src/ve_quant_kernels.c',
        'src/qwen_dense_cache.h',
        'src/nec_llama_compat.cpp', 'tests/check_qwen_accurate.cpp',
        'tests/check_qwen_nlc.cpp', 'framework/CMakeLists.txt',
        'scripts/build_qwen_accurate.sh', 'scripts/record_qwen_build.py',
        'build/qwen-cpu/qwen-infer-cpu-accurate',
        'build/qwen-cpu/llama/src/libllama.a',
        'build/qwen-cpu/llama/ggml/src/libggml.a',
        'build/qwen-cpu/llama/ggml/src/libggml-cpu.a',
        'build/qwen-cpu/llama/ggml/src/libggml-base.a',
        'build/qwen-ve-accurate/qwen-infer-ve',
        'build/qwen-ve-accurate/check-math',
        'build/qwen-ve-accurate/check-primitives',
        'build/qwen-ve-precision/manifest.json',
        'build/qwen-ve-precision/libggml-cpu.a',
        'build/qwen-ve-precision/libggml-base.a',
        'build/qwen-ve-precision/ve_quant_kernels.o',
        'build/llama-ve/src/libllama.a', 'build/llama-ve/ggml/src/libggml.a',
        'build/nec_llama_compat.o',
    ]
    cpu_flags = {}
    for target in ('qwen-accurate-math', 'qwen-infer-cpu-accurate'):
        path = ROOT / ('build/qwen-cpu/CMakeFiles/' + target + '.dir/flags.make')
        cpu_flags[target] = [line for line in path.read_text().splitlines()
                             if line.startswith(('CXX_FLAGS =', 'CXX_DEFINES ='))]
    report = dict(schema=1, recorded_utc=datetime.now(timezone.utc).isoformat(),
                  math_mode='fp64_accumulation', activation_storage='float32',
                  ve_compile_flags=args.ve_flags, ve_hook_define='QWEN_NLC',
                  cpu_compile_flags=cpu_flags,
                  artifacts=[fingerprint(path) for path in artifacts])
    output = args.output.resolve()
    output.relative_to(ROOT)  # Keep this local build record in the project.
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output)


if __name__ == '__main__':
    main()
