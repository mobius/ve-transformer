"""Capture standalone native GEMM probe binaries, flags and NLC libraries."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = ['src/ve_nlc_gemm_probe.c', 'scripts/build_nlc_gemm_probe.sh',
             'scripts/record_nlc_gemm_build.py', 'build/nlc-gemm/seq', 'build/nlc-gemm/omp']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    libraries = {name: hashlib.sha256((Path('/opt/nec/ve/nlc/3.1.0/lib')/name).read_bytes()).hexdigest()
                 for name in ('libcblas.so', 'libblas_sequential.so', 'libblas_openmp.so')}
    report = {'backend': 'native_ve', 'nlc_version': '3.1.0', 'integer_interface': 32,
              'flags': 'O1, no fast-math or associative math; OpenMP binary only uses fopenmp',
              'compiler_duty_percent': 25, 'sha256': hashes, 'library_sha256': libraries}
    (ROOT/'build/nlc-gemm/manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Standalone NLC probe build captured')


if __name__ == '__main__':
    main()
