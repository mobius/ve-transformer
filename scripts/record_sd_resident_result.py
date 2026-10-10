"""Publish verified same-binary SD-Turbo resident comparison and scope."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from record_qwen36_mtp import safe, thermal
from record_sd_im2col_result import fan_detail

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    for name in ('benchmark', 'benchmark-log', 'six', 'six-log', 'four', 'four-log', 'build-log', 'operators-log'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((ROOT/'build/sd-baseline-ve/manifest.json').read_text())
    binary = digest(ROOT/'build/sd-baseline-ve/bin/sd')
    checker = digest(ROOT/'tests/check_sd_resident.py')
    for name, value in manifest['sha256'].items():
        if digest(ROOT/name) != value:
            raise RuntimeError('current build source/binary mismatch')
    if manifest['backend'] != 've' or manifest['sha256']['build/sd-baseline-ve/bin/sd'] != binary:
        raise RuntimeError('native VE build required')

    def local(path):
        path = path.resolve()
        path.relative_to(ROOT/'build')
        return path

    def relative(path):
        return str(local(path).relative_to(ROOT))

    def verified(folder, mode, steps, cases):
        folder = local(folder)
        report = json.loads((folder/'summary.json').read_text())
        if (not report['completed'] or report['returncode'] or not report['shared_threadpool'] or
                report['mode'] != mode or report['steps'] != steps or report['threads'] != 8 or
                report['binary_sha256'] != binary or report['checker_sha256'] != checker or
                digest(folder/'check_sd_resident.py') != checker or
                [row['reference_case'] for row in report['requests']] != cases):
            raise RuntimeError('matching completed native request sequence required')
        for index, row in enumerate(report['requests']):
            loads = ['clip', 'unet', 'vae'] if mode == 'reload' or index == 0 else []
            if (row['request'] != index or row['weight_load_stages'] != loads or
                    len(row['checks']) != 3+2*steps or not all(check['passed'] and math.isfinite(check['relative_l2'])
                                                            and math.isfinite(check['max_abs_error']) for check in row['checks'])):
                raise RuntimeError('every request and numerical check must pass')
            dest = folder/('request%d' % index)
            if len(row['trace_sha256']) != 4+3*steps:
                raise RuntimeError('complete trace set required')
            for name, value in dict(row['trace_sha256'], **{'image.png': row['png_sha256']}).items():
                if digest(dest/name) != value:
                    raise RuntimeError('native artifact checksum mismatch')
        return report

    bench = json.loads((local(args.benchmark)/'summary.json').read_text())
    if (not bench['completed'] or bench['kind'] != 'sd_turbo_resident_abba' or
            bench['binary_sha256'] != binary or bench['checker_sha256'] != checker or
            bench['case_sequence'] != '0,0' or not bench['all_trace_and_png_bytes_identical'] or
            [run['mode'] for run in bench['runs']] != ['reload', 'resident', 'resident', 'reload']):
        raise RuntimeError('same-binary completed ABBA required')
    verified_runs = []
    for run in bench['runs']:
        native = verified(ROOT/run['artifacts'], run['mode'], 1, [0, 0])
        memory = json.loads((local(ROOT/run['memory_artifacts'])/'summary.json').read_text())
        if not memory['completed'] or memory['returncode'] or memory != run['memory']:
            raise RuntimeError('complete matching memory evidence required')
        if native['requests'] != run['requests'] or native['process_seconds'] != run['process_seconds']:
            raise RuntimeError('benchmark/native summary differs')
        verified_runs.append(native)
    for index in range(2):
        first = verified_runs[0]['requests'][index]
        if not all(row['requests'][index]['trace_sha256'] == first['trace_sha256'] and
                   row['requests'][index]['png_sha256'] == first['png_sha256'] for row in verified_runs):
            raise RuntimeError('actual ABBA trace and PNG equality required')
    six = verified(args.six, 'resident', 1, list(range(6)))
    four = verified(args.four, 'resident', 4, [0, 0])
    if (four['requests'][0]['trace_sha256'] != four['requests'][1]['trace_sha256'] or
            four['requests'][0]['png_sha256'] != four['requests'][1]['png_sha256']):
        raise RuntimeError('repeated four-step traces and PNG must match bytewise')
    if any(row['model_revision'] != six['model_revision'] for row in verified_runs+[four]):
        raise RuntimeError('all model revisions must match')
    for metric, request in [('process', None), ('request0', 0), ('request1', 1)]:
        def mean(mode):
            values = [row['process_seconds'] if request is None else row['requests'][request]['request_seconds']
                      for row in verified_runs if row['mode'] == mode]
            return sum(values)/len(values)
        before, after = mean('reload'), mean('resident')
        expected = {'reload_seconds': before, 'resident_seconds': after,
                    'latency_reduction_percent': (1-after/before)*100}
        if any(abs(bench['metrics'][metric][key]-value) > 1e-9 for key, value in expected.items()):
            raise RuntimeError('benchmark metrics differ from native runs')

    def additional_memory(log):
        paths = re.findall(r'VE memory samples: (build/results/[^\s]+)', local(log).read_text())
        if len(paths) != 1:
            raise RuntimeError('one complete memory trace per extra validation required')
        memory = json.loads((local(ROOT/paths[0])/'summary.json').read_text())
        if not memory['completed'] or memory['returncode'] or memory['runtime_slot'] != 1:
            raise RuntimeError('extra validation memory sampling must pass')
        return {'artifacts': paths[0], 'memory': memory}
    if 'IMAGE_OPERATOR_PASS cases=12;' not in local(args.operators_log).read_text():
        raise RuntimeError('current operator regression required')
    adopted = (bench['metrics']['process']['latency_reduction_percent'] > 0 and
               bench['metrics']['request1']['latency_reduction_percent'] > 0)
    report = {'status': 'resident_candidate_verified' if adopted else 'resident_candidate_not_adopted',
              'recommended_for_validated_workload': adopted, 'default_enabled': False,
              'model_repo': 'stabilityai/sd-turbo', 'model_revision': six['model_revision'],
              'precision': 'FP32', 'size': 512, 'runtime_slot': 1, 'build': manifest,
              'implementation': 'lazy resident CLIP/UNet/VAE weights; one shared blocking eight-thread pool; per-request output workspace and UNet scheduler cleanup; fixed SiLU/Softmax/im2col ve',
              'scope': 'ABBA two processes per arm and two identical one-step requests per process; six distinct one-step requests and two four-step requests; includes diagnostics and uncontrolled file caches',
              'production_service_verified': False, 'quality_benchmark_performed': False,
              'instantaneous_or_model_exclusive_hbm_peak_measured': False,
              'benchmark_artifacts': relative(args.benchmark), 'benchmark': bench,
              'six_artifacts': relative(args.six), 'six': six,
              'four_artifacts': relative(args.four), 'four': four,
              'six_memory': additional_memory(args.six_log), 'four_memory': additional_memory(args.four_log),
              'temperatures': {name: thermal(local(getattr(args, name))) for name in
                               ('benchmark_log', 'six_log', 'four_log', 'build_log', 'operators_log')},
              'fan_observation': fan_detail(local(args.benchmark_log))}
    safe(report)
    out = ROOT/'docs/results'/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-sd-turbo-resident.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('Public resident evidence:', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
