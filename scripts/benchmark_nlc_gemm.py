"""Serial native NLC variants on observed matrices, with independent FP64 samples."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def input_value(index, salt):
    x = (index & 0xffffffff) ^ salt
    x ^= x >> 16
    x = (x*0x7feb352d) & 0xffffffff
    x ^= x >> 15
    x = (x*0x846ca68b) & 0xffffffff
    x ^= x >> 16
    return ((x & 1023)-512)/512.0


def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED') != '1':
        raise RuntimeError('temperature supervision required')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((ROOT/'build/nlc-gemm/manifest.json').read_text())
    for name, value in manifest['sha256'].items():
        if digest(ROOT/name) != value:
            raise RuntimeError('probe build changed')
    for name, value in manifest['library_sha256'].items():
        if digest(Path('/opt/nec/ve/nlc/3.1.0/lib')/name) != value:
            raise RuntimeError('NLC library changed')
    shape_path = ROOT/'build/nlc-probe-shapes.json'
    selection = json.loads(shape_path.read_text())
    profile_path = (ROOT/selection['source_profile']).resolve()
    profile_path.relative_to(ROOT/'docs/results')
    if digest(profile_path) != selection['source_profile_sha256']:
        raise RuntimeError('shape source profile changed')
    profile = json.loads(profile_path.read_text())
    expected = []
    for stage, count in (('vae', 6), ('unet', 1), ('clip', 1)):
        expected.extend([s for s in profile['requests'][1]['shapes_by_total_seconds'] if s['stage'] == stage][:count])
    if selection['shapes'] != expected or len(expected) != 8 or profile['status'] != 'nlc_gemm_profile_verified':
        raise RuntimeError('exact verified shape selection required')
    folder = ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-nlc-gemm-probe', time.gmtime())
    folder.mkdir()
    report = {'completed': False, 'build': manifest, 'benchmark_sha256': digest(Path(__file__)),
              'source_profile': selection['source_profile'], 'source_profile_sha256': digest(profile_path),
              'shapes_sha256': digest(shape_path), 'reference': 'independent host FP64 dot products at twenty boundary/interior positions per GEMM run; not full numerical coverage',
              'absolute_tolerance': .0001, 'relative_tolerance': .0001, 'runs': []}
    (folder/'benchmark_nlc_gemm.py').write_bytes(Path(__file__).read_bytes())
    (folder/'shapes.json').write_bytes(shape_path.read_bytes())
    def save():
        (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    for index, shape in enumerate(expected):
        m, n, k, lda, ldb, ldc = (shape[key] for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldc'))
        if shape['trans_a'] != 'N' or shape['trans_b'] != 'T' or 4*(m*lda+n*ldb+m*ldc) > 4*1024**3:
            raise RuntimeError('supported actual layout and memory budget required')
        coordinates = [(0, 0), (m-1, n-1), (0, n-1), (m-1, 0)]+[((s*7919+17)%m, (s*104729+23)%n) for s in range(4, 20)]
        if len(set(coordinates)) != 20:
            raise RuntimeError('twenty distinct points required')
        reference = [math.fsum(input_value(row*lda+q, 0x12345678)*input_value(column*ldb+q, 0x9abcdef0)
                               for q in range(k)) for row, column in coordinates]
        for mode, threads in [('seq', 1), ('omp', 1), ('omp', 2), ('omp', 4), ('omp', 8)]:
            binary = ROOT/'build/nlc-gemm'/mode
            if digest(binary) != manifest['sha256'][str(binary.relative_to(ROOT))] or digest(Path(__file__)) != report['benchmark_sha256']:
                raise RuntimeError('tested executable or benchmark changed')
            env = dict(os.environ, VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib',
                       OMP_NUM_THREADS=str(threads), VE_OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = ['ve_exec', '-N', '1', str(binary)]+list(map(str, (m, n, k, lda, ldb, ldc, 1, 3)))
            print('GEMM case', index, shape['stage'], mode, 'threads', threads, flush=True)
            path = folder/('case%d-%s%d.log' % (index, mode, threads))
            started = time.perf_counter()
            with path.open('w') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter()-started
            text = path.read_text()
            config = re.findall(r'PROBE_CONFIG openmp=(\d+) actual_threads=(\d+) m=(\d+) n=(\d+) k=(\d+) lda=(\d+) ldb=(\d+) ldc=(\d+) bytes=(\d+)', text)
            if process.returncode or config != [tuple(map(str, (int(mode == 'omp'), threads, m, n, k, lda, ldb, ldc, 4*(m*lda+n*ldb+m*ldc))))]:
                raise RuntimeError('native configuration or actual parallel region differs')
            times = re.findall(r'PROBE_TIME iteration=(\d+) phase=(warmup|measure) seconds=([0-9.]+)', text)
            if len(times) != 4 or [(int(i), phase) for i, phase, _ in times] != [(0, 'warmup'), (1, 'measure'), (2, 'measure'), (3, 'measure')] or not all(math.isfinite(float(t)) and float(t) > 0 for _, _, t in times):
                raise RuntimeError('one warmup and three positive measured timings required')
            samples = re.findall(r'PROBE_SAMPLE iteration=(\d+) sample=(\d+) row=(\d+) column=(\d+) value=([^\s]+)', text)
            if len(samples) != 80:
                raise RuntimeError('twenty samples from each of four GEMMs required')
            max_error = 0.
            for position, (run, sample, row, column, value) in enumerate(samples):
                expected_run, expected_sample = divmod(position, 20)
                if (int(run), int(sample)) != (expected_run, expected_sample) or (int(row), int(column)) != coordinates[expected_sample]:
                    raise RuntimeError('independent sample ordering or location differs')
                actual = float(value)
                error = abs(actual-reference[expected_sample])
                if not math.isfinite(actual) or error > .0001+.0001*abs(reference[expected_sample]):
                    raise RuntimeError('FP64 sample numerical check failed')
                max_error = max(max_error, error)
            if re.findall(r'PROBE_FINITE elements=(\d+) PASS', text) != [str(m*n)]:
                raise RuntimeError('full final output finite check required')
            measurements = [float(t) for _, phase, t in times if phase == 'measure']
            report['runs'].append({'shape_index': index, 'shape': shape, 'mode': mode, 'requested_threads': threads,
                                   'parallel_region_threads': threads, 'process_seconds': elapsed, 'warmup_seconds': float(times[0][2]),
                                   'kernel_seconds': measurements, 'mean_kernel_seconds': sum(measurements)/3,
                                   'sample_checks_passed': 80, 'max_abs_sample_error': max_error,
                                   'final_output_all_finite': True, 'raw_log_sha256': digest(path)})
            save()
            print('PASS kernel mean', round(sum(measurements)/3, 6), 'FP64 samples', 80, flush=True)
    report.update(completed=True, scope='eight observed shapes, sequential versus OpenMP1/2/4/8; three kernel measurements per variant, fixed synthetic FP32 data; parallel region verifies runtime capacity, not per-kernel physical utilization; no full-graph or production speed claim')
    save()
    print('NLC GEMM verified:', folder.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
