"""Serial ABBA of frozen sequential/pthread and unified OpenMP VE builds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(path, unified):
    path = path.resolve()
    path.relative_to(ROOT/'build')
    manifest = path/'build/sd-baseline-ve/manifest.json'
    value = json.loads(manifest.read_text())
    if value.get('nlc_mode') != ('openmp' if unified else 'sequential'):
        raise RuntimeError('incorrect NLC arm')
    if value.get('generic_thread_runtime') != ('openmp' if unified else 'pthread'):
        raise RuntimeError('incorrect generic runtime arm')
    for name, digest in value['sha256'].items():
        source = (path/name).resolve()
        source.relative_to(path)
        if sha(source) != digest:
            raise RuntimeError('snapshot checksum mismatch')
    return path, value, sha(manifest)


def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED') != '1':
        raise RuntimeError('temperature supervision required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.resolve()
    reference.relative_to(ROOT/'build')
    baseline = verify_snapshot(args.baseline, False)
    candidate = verify_snapshot(args.candidate, True)
    # Only the executable may differ in the archived source/generated inputs.
    sources = lambda value: {k: v for k, v in value['sha256'].items() if k != 'build/sd-baseline-ve/bin/sd'}
    if sources(baseline[1]) != sources(candidate[1]):
        raise RuntimeError('source inputs differ between runtime arms')
    checker = ROOT/'tests/check_sd_resident.py'
    libraries = [Path('/opt/nec/ve/nlc/3.1.0/lib')/name for name in
                 ('libcblas.so', 'libblas_sequential.so', 'libblas_openmp.so')]
    library_sha = {p.name: sha(p) for p in libraries}
    folder = ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-runtime-abba', time.gmtime())
    folder.mkdir()
    report = {'completed': False, 'comparison': 'sequential pthread versus unified OpenMP eight',
              'same_binary': False, 'checker_sha256': sha(checker), 'benchmark_sha256': sha(Path(__file__)),
              'reference_artifacts': str(reference.relative_to(ROOT)), 'library_sha256': library_sha,
              'builds': {name: {'snapshot': str(arm[0].relative_to(ROOT)), 'manifest_sha256': arm[2],
                               'binary_sha256': arm[1]['sha256']['build/sd-baseline-ve/bin/sd']}
                         for name, arm in (('baseline', baseline), ('candidate', candidate))}, 'runs': []}
    def save():
        (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    (folder/'benchmark_sd_runtime.py').write_bytes(Path(__file__).read_bytes())
    for index, name in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
        arm = baseline if name == 'baseline' else candidate
        if sha(checker) != report['checker_sha256'] or {p.name: sha(p) for p in libraries} != library_sha:
            raise RuntimeError('checker or libraries changed')
        if verify_snapshot(arm[0], name == 'candidate')[2] != arm[2]:
            raise RuntimeError('manifest changed')
        print('Runtime ABBA', index, name, flush=True)
        command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/sample_ve_memory.py'), '--timeout', '300', '--',
                   str(ROOT/'.venv/bin/python'), str(checker), '--build-snapshot', str(arm[0]),
                   '--reference', str(reference), '--mode', 'resident', '--tokenizer', 'resident',
                   '--nlc-threads', 'single' if name == 'baseline' else 'unified', '--cases', '0,0']
        path = folder/('run%d.log' % index)
        with path.open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        text = path.read_text()
        native = re.findall(r'Native sequential requests verified: (build/results/[^\s]+)', text)
        samples = re.findall(r'VE memory samples: (build/results/[^\s]+)', text)
        if result.returncode or len(native) != 1 or len(samples) != 1:
            raise RuntimeError('completed native and memory evidence required')
        run = json.loads((ROOT/native[0]/'summary.json').read_text())
        memory = json.loads((ROOT/samples[0]/'summary.json').read_text())
        if not run['completed'] or not memory['completed'] or memory['final_used_kib'] != 131072:
            raise RuntimeError('completion and memory recovery required')
        if run['binary_sha256'] != report['builds'][name]['binary_sha256'] or run['checker_sha256'] != report['checker_sha256']:
            raise RuntimeError('binary or checker evidence differs')
        if run['steps'] != 1 or len(run['requests']) != 2 or not all(len(r['checks']) == 5 and all(c['passed'] for c in r['checks']) for r in run['requests']):
            raise RuntimeError('every independent CPU stage check must pass')
        report['runs'].append({'arm': name, 'artifacts': native[0], 'memory_artifacts': samples[0],
                               'process_seconds': run['process_seconds'], 'requests': run['requests'], 'memory': memory})
        save()
    metrics = {}
    for label, index in (('process', None), ('cold', 0), ('hot', 1)):
        means = {name: sum(r['process_seconds'] if index is None else r['requests'][index]['request_seconds']
                          for r in report['runs'] if r['arm'] == name)/2 for name in ('baseline', 'candidate')}
        metrics[label] = {**means, 'latency_reduction_percent': (1-means['candidate']/means['baseline'])*100}
    report.update(completed=True, metrics=metrics,
                  all_trace_and_png_bytes_identical=all(
                      r['requests'][i]['trace_sha256'] == report['runs'][0]['requests'][i]['trace_sha256'] and
                      r['requests'][i]['png_sha256'] == report['runs'][0]['requests'][i]['png_sha256']
                      for r in report['runs'] for i in (0, 1)),
                  scope='two runs per build; one fixed prompt and one Euler step; includes diagnostics; runtime and NLC jointly changed')
    save()
    print('Runtime ABBA completed:', folder.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
