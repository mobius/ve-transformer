"""Serial same-binary ABBA of SD-Turbo reuse or operator/thread configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED') != '1':
        raise RuntimeError('temperature supervision required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cases', default='0,0')
    parser.add_argument('--comparison', choices=('weights', 'tokenizer', 'nlc', 'binary', 'vae', 'vae_selective', 'gelu', 'im2col', 'im2col_extend', 'gemm_spatial', 'png','rgb_buffer','pixels'), default='weights')
    args = parser.parse_args()
    reference = args.reference.resolve()
    reference.relative_to(ROOT/'build')
    suffix = {'weights': 'sd-resident-abba', 'tokenizer': 'sd-tokenizer-abba', 'nlc': 'sd-nlc-abba', 'binary': 'sd-binary-abba', 'vae': 'sd-vae-abba', 'vae_selective': 'sd-vae-selective-abba', 'gelu': 'sd-gelu-abba', 'im2col': 'sd-im2col-abba', 'im2col_extend': 'sd-im2col-extend-abba', 'gemm_spatial':'sd-gemm-spatial-abba','png':'sd-png-abba','rgb_buffer':'sd-rgb-buffer-abba','pixels':'sd-pixels-abba'}[args.comparison]
    folder = ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-'+suffix, time.gmtime())
    folder.mkdir()
    binary = ROOT/'build/sd-baseline-ve/bin/sd'
    checker = ROOT/'tests/check_sd_resident.py'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'completed': False, 'kind': {'weights': 'sd_turbo_resident_abba', 'tokenizer': 'sd_turbo_tokenizer_abba', 'nlc': 'sd_turbo_nlc_abba', 'binary': 'sd_turbo_binary_abba', 'vae': 'sd_turbo_vae_abba', 'vae_selective': 'sd_turbo_vae_selective_abba', 'gelu': 'sd_turbo_gelu_abba', 'im2col': 'sd_turbo_im2col_abba', 'im2col_extend': 'sd_turbo_im2col_extend_abba', 'gemm_spatial':'sd_turbo_gemm_spatial_abba','png':'sd_turbo_png_abba','rgb_buffer':'sd_turbo_rgb_buffer_abba','pixels':'sd_turbo_pixels_abba'}[args.comparison],
              'comparison': args.comparison,
              'binary_sha256': digest(binary), 'checker_sha256': digest(checker),
              'benchmark_sha256': digest(Path(__file__)), 'case_sequence': args.cases,
              'reference_artifacts': str(reference.relative_to(ROOT)), 'runs': []}

    def save():
        (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    (folder/'benchmark_sd_resident.py').write_bytes(Path(__file__).read_bytes())
    arms = {'weights': ('reload', 'resident', 'resident', 'reload'), 'tokenizer': ('request', 'resident', 'resident', 'request'), 'nlc': ('single', 'staged', 'staged', 'single'), 'binary': ('generic','ve','ve','generic'), 'vae': ('8','4','4','8'), 'vae_selective': ('all4','selective','selective','all4'), 'gelu': ('generic','ve','ve','generic'), 'im2col': ('channels','rows','rows','channels'), 'im2col_extend': ('rows','rows_256','rows_256','rows'), 'gemm_spatial':('none','8192','8192','none'),'png':('generic','ve','ve','generic'),'rgb_buffer':('request','resident','resident','request'),'pixels':('generic','ve','ve','generic')}[args.comparison]
    for index, arm in enumerate(arms):
        mode = arm if args.comparison == 'weights' else 'resident'
        tokenizer = 'request' if args.comparison == 'weights' else 'resident' if args.comparison in ('nlc','binary','vae','vae_selective','gelu','im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels') else arm
        nlc_threads = arm if args.comparison == 'nlc' else 'unified' if args.comparison in ('binary','vae','vae_selective','gelu','im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels') else 'single'
        binary_scalar = arm if args.comparison=='binary' else 've' if args.comparison in ('vae','vae_selective','gelu','im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels') else 'generic'
        vae_threads = (4 if arm=='all4' else 8) if args.comparison=='vae_selective' else int(arm) if args.comparison=='vae' else 8
        vae_blas_threads = 4 if args.comparison in ('gelu','im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels') or (args.comparison=='vae_selective' and arm=='selective') else None
        gelu = arm if args.comparison=='gelu' else 've' if args.comparison in ('im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels') else 'generic'
        im2col_mode = 'rows_256' if args.comparison in ('gemm_spatial','png','rgb_buffer','pixels') else arm if args.comparison in ('im2col','im2col_extend') else 'channels'
        vae_spatial_tile='8192' if args.comparison in ('png','rgb_buffer','pixels') else arm if args.comparison=='gemm_spatial' else 'none'
        png_encoder=arm if args.comparison=='png' else 've' if args.comparison in ('rgb_buffer','pixels') else 'generic'
        rgb_buffer=arm if args.comparison=='rgb_buffer' else 'resident' if args.comparison=='pixels' else 'request'
        pixel_kernel=arm if args.comparison=='pixels' else 'generic'
        if digest(binary) != report['binary_sha256'] or digest(checker) != report['checker_sha256']:
            raise RuntimeError('binary or checker changed during comparison')
        command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/sample_ve_memory.py'), '--',
                   str(ROOT/'.venv/bin/python'), str(checker), '--reference', str(reference),
                   '--mode', mode, '--tokenizer', tokenizer, '--nlc-threads', nlc_threads, '--binary-scalar', binary_scalar, '--vae-threads', str(vae_threads), '--gelu', gelu, '--cases', args.cases, '--im2col-mode', im2col_mode, '--vae-spatial-tile', vae_spatial_tile, '--png-encoder', png_encoder, '--rgb-buffer', rgb_buffer, '--pixel-kernel', pixel_kernel]
        if vae_blas_threads is not None:
            command.extend(['--vae-blas-threads', str(vae_blas_threads)])
        path = folder/('run%d.log' % index)
        print('ABBA', args.comparison, index, arm, flush=True)
        with path.open('w') as log:
            # The sampler owns timeout/descendant cleanup. Killing only that
            # wrapper via subprocess.run(timeout=...) could orphan its group.
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        text = path.read_text()
        native = re.findall(r'Native sequential requests verified: (build/results/[^\s]+)', text)
        sampled = re.findall(r'VE memory samples: (build/results/[^\s]+)', text)
        if result.returncode or len(native) != 1 or len(sampled) != 1:
            raise RuntimeError('completed native and memory evidence required')
        run = json.loads((ROOT/native[0]/'summary.json').read_text())
        memory = json.loads((ROOT/sampled[0]/'summary.json').read_text())
        if (not run['completed'] or not run.get('shared_threadpool') or not memory['completed'] or run['mode'] != mode or run.get('tokenizer_mode') != tokenizer or run.get('nlc_threads') != nlc_threads or
                (args.comparison == 'nlc' and run.get('nlc_mode') != 'openmp') or
                (args.comparison in ('binary','vae','vae_selective','gelu','im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels') and (run.get('binary_scalar_mode')!=binary_scalar or run.get('nlc_mode')!='openmp')) or
                run.get('pixel_kernel') != pixel_kernel or run.get('rgb_buffer') != rgb_buffer or run.get('png_encoder') != png_encoder or run.get('image_output_profile_enabled') or run.get('vae_spatial_tile') != vae_spatial_tile or run.get('vae_threads') != vae_threads or run.get('vae_blas_threads') != vae_blas_threads or run.get('gelu_mode') != gelu or
                run['binary_sha256'] != report['binary_sha256'] or run['checker_sha256'] != report['checker_sha256']):
            raise RuntimeError('matching verified run required')
        if not all(len(row['checks']) == 3+2*run['steps'] and all(check['passed'] for check in row['checks'])
                   for row in run['requests']):
            raise RuntimeError('every independent numerical check must pass')
        if args.comparison in ('im2col','im2col_extend','gemm_spatial','png','rgb_buffer','pixels'):
            for row in run['requests']:
                dispatch=row.get('im2col_row_dispatch',[])
                if not dispatch or any(item['enabled']!=(im2col_mode!='channels') for item in dispatch):
                    raise RuntimeError('missing/mismatched im2col row dispatch')
        report['runs'].append({'mode': arm, 'pixel_kernel':pixel_kernel, 'rgb_buffer':rgb_buffer, 'png_encoder':png_encoder,'vae_spatial_tile':vae_spatial_tile, 'im2col_mode': im2col_mode, 'weight_mode': mode, 'tokenizer_mode': tokenizer, 'nlc_threads': nlc_threads, 'binary_scalar_mode': binary_scalar, 'vae_threads': vae_threads, 'vae_blas_threads': vae_blas_threads, 'gelu_mode': gelu, 'artifacts': native[0], 'memory_artifacts': sampled[0],
                               'process_seconds': run['process_seconds'], 'requests': run['requests'], 'memory': memory})
        save()
    if digest(binary) != report['binary_sha256'] or digest(checker) != report['checker_sha256']:
        raise RuntimeError('binary or checker changed during comparison')
    runs = report['runs']
    byte_identical = True
    for index in range(len(runs[0]['requests'])):
        first = runs[0]['requests'][index]
        identical = all(run['requests'][index]['trace_sha256'] == first['trace_sha256'] and
                        run['requests'][index]['png_sha256'] == first['png_sha256'] for run in runs)
        byte_identical = byte_identical and identical
        if not identical and args.comparison not in ('nlc','vae','vae_selective','gelu'):
            raise RuntimeError('reuse must preserve every trace and PNG byte')
    def average(mode, request=None):
        values = [run['process_seconds'] if request is None else run['requests'][request]['request_seconds']
                  for run in runs if run['mode'] == mode]
        return sum(values)/len(values)
    metrics = {}
    for key, request in [('process', None)]+[('request%d' % i, i) for i in range(len(runs[0]['requests']))]:
        baseline = {'weights': 'reload', 'tokenizer': 'request', 'nlc': 'single', 'binary':'generic', 'vae':'8', 'vae_selective':'all4', 'gelu':'generic', 'im2col':'channels', 'im2col_extend':'rows','gemm_spatial':'none','png':'generic','rgb_buffer':'request','pixels':'generic'}[args.comparison]
        candidate = 've' if args.comparison in ('png','pixels') else '8192' if args.comparison=='gemm_spatial' else 'rows_256' if args.comparison=='im2col_extend' else 'rows' if args.comparison=='im2col' else 've' if args.comparison=='gelu' else 'selective' if args.comparison=='vae_selective' else '4' if args.comparison=='vae' else 'staged' if args.comparison == 'nlc' else 've' if args.comparison=='binary' else 'resident'
        before, after = average(baseline, request), average(candidate, request)
        metrics[key] = { {'weights': 'reload_seconds', 'tokenizer': 'request_seconds', 'nlc': 'single_seconds', 'binary':'generic_seconds', 'vae':'eight_seconds', 'vae_selective':'all_four_seconds', 'gelu':'generic_seconds', 'im2col':'channels_seconds', 'im2col_extend':'rows512_seconds','gemm_spatial':'none_seconds','png':'generic_seconds','rgb_buffer':'request_seconds','pixels':'generic_seconds'}[args.comparison]: before, ('ve_seconds' if args.comparison in ('png','pixels') else 'tile_seconds' if args.comparison=='gemm_spatial' else 'rows256_seconds' if args.comparison=='im2col_extend' else 'rows_seconds' if args.comparison=='im2col' else 've_seconds' if args.comparison=='gelu' else 'selective_seconds' if args.comparison=='vae_selective' else 'four_seconds' if args.comparison=='vae' else 'staged_seconds' if args.comparison == 'nlc' else 've_seconds' if args.comparison=='binary' else 'resident_seconds'): after, 'latency_reduction_percent': (1-after/before)*100}
    report.update(completed=True, metrics=metrics, all_trace_and_png_bytes_identical=byte_identical,
                  decision='faster_pending_multi_prompt_validation' if metrics['process']['latency_reduction_percent'] > 0 else 'not_faster',
                  scope='two same-process requests by default, two runs per arm, includes diagnostics; no production service or arbitrary prompt claim')
    save()
    print('ABBA completed:', folder.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
