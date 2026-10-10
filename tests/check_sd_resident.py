"""Verify sequential native SD-Turbo requests and actual weight reuse on VE."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from export_sd_reference import PROMPTS
from check_sd_turbo import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED') != '1':
        raise RuntimeError('temperature supervision required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--build-snapshot', type=Path, help='immutable build snapshot under build/')
    parser.add_argument('--mode', choices=('reload', 'resident'), required=True)
    parser.add_argument('--cases', default='0,0', help='ordered reference case IDs, two to sixteen requests')
    parser.add_argument('--tokenizer', choices=('request', 'resident'), default='request')
    parser.add_argument('--nlc-threads', choices=('single', 'staged', 'unified'), default='single')
    parser.add_argument('--op-profile', action='store_true', help='diagnostic operator timing; not a performance comparison')
    parser.add_argument('--binary-shape-profile', action='store_true')
    parser.add_argument('--im2col-shape-profile', action='store_true')
    parser.add_argument('--binary-scalar', choices=('generic','ve'), default='generic')
    parser.add_argument('--vae-threads', type=int, choices=(4,8), default=8)
    parser.add_argument('--vae-blas-threads', type=int, choices=(4,), default=None, help='experimental matrix-only four threads; generic must remain eight')
    parser.add_argument('--gelu', choices=('generic','ve'), default='generic')
    parser.add_argument('--im2col-mode', choices=('channels','rows','rows_256'), default='channels')
    parser.add_argument('--vae-pretranspose', action='store_true')
    parser.add_argument('--vae-spatial-tile', choices=('none','8192','mixed4096','mixed2048'), default='none')
    parser.add_argument('--image-output-profile', action='store_true')
    parser.add_argument('--png-encoder', choices=('generic','ve'), default='generic')
    parser.add_argument('--rgb-buffer', choices=('request','resident'), default='request')
    parser.add_argument('--pixel-kernel', choices=('generic','ve'), default='generic')
    args = parser.parse_args()
    if args.binary_shape_profile and not args.op_profile:
        parser.error('binary shape profiling requires operator profiling')
    if args.vae_threads==4 and args.nlc_threads!='unified':
        parser.error('VAE four threads require unified OpenMP')
    if args.vae_blas_threads is not None and (args.vae_threads!=8 or args.nlc_threads!='unified'):
        parser.error('selective VAE requires generic eight and unified OpenMP')
    sequence = [int(value) for value in args.cases.split(',')]
    if not 2 <= len(sequence) <= 16:
        raise RuntimeError('two to sixteen requests required')
    reference = args.reference.resolve()
    reference.relative_to(ROOT/'build')
    cpu = json.loads((reference/'summary.json').read_text())
    if not cpu['completed'] or cpu['size'] != 512 or cpu['steps'] not in (1, 4):
        raise RuntimeError('completed supported CPU reference required')
    model = ROOT/'build/models/sd-turbo'
    provenance = json.loads((model/'provenance.json').read_text())
    if not provenance['completed'] or provenance['revision'] != cpu['model_revision']:
        raise RuntimeError('model provenance mismatch')
    for name, item in provenance['files'].items():
        if sha(model/name) != item['sha256']:
            raise RuntimeError('official weight checksum mismatch')
    build_root = args.build_snapshot.resolve() if args.build_snapshot else ROOT
    if args.build_snapshot:
        build_root.relative_to(ROOT/'build')
    manifest = json.loads((build_root/'build/sd-baseline-ve/manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        source = (build_root/name).resolve()
        source.relative_to(build_root)
        if sha(source) != digest:
            raise RuntimeError('build checksum mismatch')
    nlc_mode = manifest.get('nlc_mode', 'sequential')
    unified = args.nlc_threads == 'unified'
    if (manifest.get('generic_thread_runtime','pthread') == 'openmp') != unified:
        raise RuntimeError('generic OpenMP build requires unified fixed-thread validation')
    if nlc_mode != 'openmp' and args.nlc_threads != 'single':
        raise RuntimeError('staged threads require an OpenMP NLC build')
    binary = build_root/'build/sd-baseline-ve/bin/sd'
    folder = ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-resident', time.gmtime())
    folder.mkdir()
    rows = {row['case']: row for row in cpu['cases']}
    requests = []
    lines = []
    for index, case in enumerate(sequence):
        if case not in rows:
            raise RuntimeError('requested CPU reference missing')
        row = rows[case]
        source = reference/('case%d' % case)
        for name, item in row['arrays'].items():
            if sha(source/(name+'.f32')) != item['sha256']:
                raise RuntimeError('reference array checksum mismatch')
        dest = folder/('request%d' % index)
        dest.mkdir()
        fields = [PROMPTS[case], str(source/'noise.f32'), str(source/'sigmas.f32'), str(dest), str(dest/'image.png')]
        if any('\t' in value or '\n' in value or '\r' in value for value in fields):
            raise RuntimeError('invalid request manifest field')
        lines.append('\t'.join(fields))
        requests.append({'request': index, 'reference_case': case, 'checks': []})
    request_file = folder/'requests.tsv'
    request_file.write_text('\n'.join(lines)+'\n')
    report = {'vae_pretranspose':args.vae_pretranspose,'pixel_kernel':args.pixel_kernel,'rgb_buffer':args.rgb_buffer,'png_encoder':args.png_encoder,'image_output_profile_enabled':args.image_output_profile,'vae_spatial_tile':args.vae_spatial_tile, 'im2col_mode': args.im2col_mode, 'completed': False, 'mode': args.mode, 'steps': cpu['steps'], 'threads': 8,
              'settings': 'one shared scheduling pool, SiLU/Softmax/im2col ve, stage profiling',
              'operator_profile_enabled': args.op_profile,
              'binary_shape_profile_enabled': args.binary_shape_profile,
              'binary_scalar_mode': args.binary_scalar, 'gelu_mode': args.gelu,
              'vae_threads': args.vae_threads,
              'shared_threadpool': True, 'tokenizer_mode': args.tokenizer, 'nlc_mode': nlc_mode, 'nlc_threads': args.nlc_threads, 'vae_blas_threads': args.vae_blas_threads,
              'binary_sha256': sha(binary), 'checker_sha256': sha(Path(__file__)),
              'model_revision': cpu['model_revision'], 'reference_artifacts': str(reference.relative_to(ROOT)),
              'requests': requests}

    def save():
        (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    (folder/'check_sd_resident.py').write_bytes(Path(__file__).read_bytes())
    env = dict(os.environ, VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib',
               OMP_NUM_THREADS='8' if unified else '1', VE_OMP_NUM_THREADS='8' if unified else '1',
               SD_NLC_FIXED_THREADS='8' if unified else '0', SD_NLC_STAGE_THREADS='1' if args.nlc_threads == 'staged' else '0',
               SD_NLC_THREAD_PROFILE='1' if nlc_mode == 'openmp' else '0', SD_NLC_PROFILE=os.environ.get('SD_NLC_PROFILE', '0'), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               SD_REUSE_TOKENIZER='1' if args.tokenizer == 'resident' else '0', SD_REQUEST_MANIFEST=str(request_file), SD_RESIDENT_WEIGHTS='1' if args.mode == 'resident' else '0',
               SD_FIXED_NOISE=str(reference/('case%d' % sequence[0])/'noise.f32'),
               SD_FIXED_SIGMAS=str(reference/('case%d' % sequence[0])/'sigmas.f32'), SD_TRACE_DIR=str(folder/'request0'),
               GGML_SCHED_DEBUG='2', SD_PROFILE='1', SD_OP_PROFILE='1' if args.op_profile else '0', SD_VE_SILU='1', SD_VE_GELU='1' if args.gelu=='ve' else '0',
               SD_BINARY_SHAPE_PROFILE='1' if args.binary_shape_profile else '0',
               SD_IM2COL_SHAPE_PROFILE='1' if args.im2col_shape_profile else '0',
               SD_VE_BINARY_SCALAR='1' if args.binary_scalar=='ve' else '0',
               SD_VE_PIXELS='1' if args.pixel_kernel=='ve' else '0', SD_REUSE_RGB_BUFFER='1' if args.rgb_buffer=='resident' else '0', SD_VE_PNG='1' if args.png_encoder=='ve' else '0', SD_IMAGE_OUTPUT_PROFILE='1' if args.image_output_profile else '0', SD_NLC_VAE_PRETRANSPOSE='1' if args.vae_pretranspose else '0', SD_NLC_VAE_SPATIAL_TILE=args.vae_spatial_tile, SD_NLC_VAE_THREADS=str(args.vae_threads), SD_NLC_VAE_BLAS_THREADS='4' if args.vae_blas_threads==4 else '0',
               SD_VE_IM2COL_ROWS='1' if args.im2col_mode!='channels' else '0',
               SD_VE_IM2COL_256='1' if args.im2col_mode=='rows_256' else '0',
               SD_VE_SOFTMAX='1', SD_VE_IM2COL='1', SD_PERSISTENT_THREADPOOL='1', SD_SHARED_THREADPOOL='1', SD_TURBO_EPS='1')
    command = ['ve_exec', '-N', '1', str(binary), '-m', str(model), '-p', PROMPTS[sequence[0]],
               '-W', '512', '-H', '512', '--steps', str(cpu['steps']), '--cfg-scale', '1', '--clip-skip', '2',
               '--sampling-method', 'euler', '--schedule', 'discrete', '--type', 'f32', '-t', '8',
               '-s', str(rows[sequence[0]]['seed']), '-o', str(folder/'request0/image.png')]
    started = time.perf_counter()
    with (folder/'native.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    report.update(returncode=result.returncode, process_seconds=time.perf_counter()-started)
    save()
    if result.returncode:
        raise RuntimeError('native multi-request run failed; inspect local log')
    log = (folder/'native.log').read_text()
    if (log.count('SD_SHARED_POOL created=1 threads=8 poll=0') != 1 or
            log.count('SD_SHARED_POOL released=1 references=0') != 1):
        raise RuntimeError('exactly one shared pool creation and balanced release required')
    segments = re.findall(r'SD_REQUEST_BEGIN index=(\d+) resident=(\d+)\n(.*?)SD_REQUEST_END index=(\d+) seconds=([0-9.]+)', log, re.S)
    if len(segments) != len(requests) or 'SD_TURBO_NATIVE_COMPLETE' not in log:
        raise RuntimeError('complete request markers required')
    for index, segment in enumerate(segments):
        entry = requests[index]
        begin, mode, body, end, elapsed = segment
        if int(begin) != index or int(end) != index or int(mode) != (args.mode == 'resident') or float(elapsed) <= 0:
            raise RuntimeError('request ordering or residency mismatch')
        loads = re.findall(r'SD_WEIGHT_LOAD request=(\d+) stage=(clip|unet|vae)', body)
        expected = [(str(index), stage) for stage in ('clip', 'unet', 'vae')] if args.mode == 'reload' or index == 0 else []
        if loads != expected:
            raise RuntimeError('actual component weight loading differs from requested lifecycle')
        weight_profiles = re.findall(r'SD_PROFILE stage=(clip|unet|vae) part=weights_load ', body)
        if weight_profiles != [stage for _, stage in expected]:
            raise RuntimeError('weight timing evidence differs from load markers')
        pools = re.findall(r'SD_THREADPOOL stage=(clip|unet|vae) threads=8 poll=0 shared=1', body)
        if pools != [stage for _, stage in expected]:
            raise RuntimeError('shared pool attachments differ from component lifecycle')
        packing = re.findall(r'SD_IM2COL stage=(clip|unet|vae) optimized_nodes=(\d+) fallback_nodes=(\d+)', body)
        if (not all(any(stage == component and int(nodes) > 0 for stage, nodes, _ in packing)
                    for component in ('unet', 'vae')) or any(int(fallback) for _, _, fallback in packing)):
            raise RuntimeError('actual im2col candidate required in every request')
        row_dispatch = re.findall(r'SD_IM2COL_ROWS stage=(clip|unet|vae) rows=(\d+) channels=(\d+) enabled=(0|1)', body)
        if args.im2col_mode!='channels' and not row_dispatch:
            raise RuntimeError('missing output-row dispatch evidence')
        if row_dispatch:
            if [r[0] for r in row_dispatch] != [r[0] for r in packing]:
                raise RuntimeError('im2col dispatch stage mismatch')
            for (stage, rows_count, channels, enabled), (_, nodes, fallback) in zip(row_dispatch, packing):
                expected_rows=(15 if args.im2col_mode=='rows_256' else 8) if args.im2col_mode!='channels' and stage=='vae' else 0
                if int(rows_count)!=expected_rows or int(rows_count)+int(channels)!=int(nodes) or int(enabled)!=(args.im2col_mode!='channels'):
                    raise RuntimeError('unexpected output-row candidate coverage')
        scope=re.findall(r'SD_IM2COL_SCOPE stage=(clip|unet|vae) extended=(0|1)',body)
        if args.im2col_mode=='rows_256' and not scope:
            raise RuntimeError('missing extended shape scope')
        if scope and ([r[0] for r in scope]!=[r[0] for r in packing] or any(int(r[1])!=(args.im2col_mode=='rows_256') for r in scope)):
            raise RuntimeError('im2col scope flag mismatch')
        entry['im2col_scope']=[dict(stage=stage,extended=int(extended)) for stage,extended in scope]
        entry['im2col_row_dispatch']=[dict(stage=stage, rows=int(rows_count), channels=int(channels), enabled=int(enabled)) for stage,rows_count,channels,enabled in row_dispatch]
        tokenizer_markers = re.findall(r'SD_TOKENIZER reuse=(0|1) initialized=(0|1)', body)
        expected_tokenizer = ('1', '1' if index == 0 or args.mode == 'reload' else '0') if args.tokenizer == 'resident' else ('0', '1')
        if tokenizer_markers != [expected_tokenizer]:
            raise RuntimeError('actual tokenizer construction/reuse marker mismatch')
        profiles = re.findall(r'SD_PROFILE stage=(clip|unet|vae) part=([a-zA-Z_]+) seconds=([0-9.]+) calls=(\d+)', body)
        if not all(any(stage == component and part == operation for stage, part, _, _ in profiles)
                   for component in ('clip', 'unet', 'vae')
                   for operation in ('graph_build', 'graph_alloc', 'graph_compute', 'backend_CPU', 'backend_BLAS')):
            raise RuntimeError('complete neural-stage profiling required per request')
        thread_markers = re.findall(r'SD_NLC_THREADS stage=(clip|unet|vae) configured=(\d+) actual=(\d+) idle_after=(\d+)', body)
        dispatch = re.findall(r'SD_BINARY_DISPATCH stage=(clip|unet|vae) op=(ADD|MUL) optimized=(\d+) fallback=(\d+) enabled=(0|1)',body)
        if args.binary_scalar=='ve':
            wanted={('clip','ADD'):(0,232),('clip','MUL'):(0,47),('unet','ADD'):(181,238),('unet','MUL'):(61,64),('vae','ADD'):(70,15),('vae','MUL'):(30,0)}
            for (component,op),(optimized,fallback) in wanted.items():
                dispatch_rows=[row for row in dispatch if row[0]==component and row[1]==op]
                count=cpu['steps'] if component=='unet' else 1
                if len(dispatch_rows)!=count or any(tuple(map(int,row[2:]))!=(optimized,fallback,1) for row in dispatch_rows):
                    raise RuntimeError('actual scalar broadcast dispatch differs')
        elif any(int(row[2]) or int(row[4]) for row in dispatch):
            raise RuntimeError('generic mode used scalar broadcast candidate')
        entry['binary_dispatch']=[{'stage':a,'op':b,'optimized':int(c),'fallback':int(d),'enabled':bool(int(e))} for a,b,c,d,e in dispatch]
        gelu_rows = re.findall(r'SD_GELU_DISPATCH stage=(clip|unet|vae) optimized=(\d+) fallback=(\d+) enabled=(0|1)', body)
        if args.gelu=='ve' or gelu_rows:
            for component,count in (('clip',23),('unet',16),('vae',0)):
                rows_for_stage = [r for r in gelu_rows if r[0]==component]
                calls = cpu['steps'] if component=='unet' else 1
                wanted = (count,0,1) if args.gelu=='ve' else (0,count,0)
                if len(rows_for_stage)!=calls or any(tuple(map(int,r[1:]))!=wanted for r in rows_for_stage):
                    raise RuntimeError('actual GELU dispatch differs')
        entry['gelu_dispatch'] = [{'stage':a,'optimized':int(b),'fallback':int(c),'enabled':bool(int(d))} for a,b,c,d in gelu_rows]
        from collections import Counter
        tiled=re.findall(r'SD_NLC_SPATIAL_TILE stage=(vae) m=(\d+) n=(\d+) k=(\d+) tile=(\d+) calls=(\d+)',body)
        expected_tiles=Counter({('vae',256,262144,2304,8192,32):1,('vae',512,65536,4608,8192,8):1,('vae',256,65536,2304,8192,8):5,('vae',256,65536,4608,8192,8):1,('vae',128,262144,2304,8192,32):1}) if args.vae_spatial_tile=='8192' else Counter()
        if args.vae_spatial_tile in ('mixed4096','mixed2048'):
            expected_tiles=Counter({('vae',256,262144,2304,8192,32):1,('vae',512,65536,4608,8192,8):1,('vae',256,65536,2304,4096,16):5,('vae',256,65536,4608,4096,16):1,('vae',128,262144,2304,8192,32):1})
        if args.vae_spatial_tile=='mixed2048':
            expected_tiles[('vae',512,16384,4608,2048,8)]=7
        if Counter((r[0],)+tuple(map(int,r[1:])) for r in tiled)!=expected_tiles or len(tiled)!=body.count('SD_NLC_SPATIAL_TILE '):
            raise RuntimeError('spatial tiling dispatch differs')
        entry['vae_spatial_dispatch']=[dict(stage=r[0],m=int(r[1]),n=int(r[2]),k=int(r[3]),tile=int(r[4]),calls=int(r[5])) for r in tiled]
        packed=re.findall(r'SD_NLC_PRETRANSPOSE stage=(vae) m=(\d+) n=(\d+) k=(\d+) tile=(\d+) calls=(\d+) bytes=(\d+)',body)
        expected_packed=Counter({('vae',512,16384,4608,2048,8,37748736):7,('vae',256,65536,2304,4096,16,37748736):5,('vae',256,65536,4608,4096,16,75497472):1}) if args.vae_pretranspose else Counter()
        if args.vae_pretranspose and (args.vae_spatial_tile!='mixed2048' or args.vae_blas_threads!=4 or args.vae_threads!=8):
            raise RuntimeError('pretranspose validation requires fixed selective mixed2048 workload')
        if Counter((r[0],)+tuple(map(int,r[1:])) for r in packed)!=expected_packed or len(packed)!=body.count('SD_NLC_PRETRANSPOSE '):raise RuntimeError('actual pretranspose dispatch differs')
        entry['vae_pretranspose_dispatch']=[dict(stage=r[0],m=int(r[1]),n=int(r[2]),k=int(r[3]),tile=int(r[4]),calls=int(r[5]),bytes=int(r[6])) for r in packed]

        png_rows=re.findall(r'SD_PNG_ENCODER mode=(generic|ve) width=(\d+) height=(\d+) channels=(\d+) stride=(\d+)',body)
        if png_rows!=[(args.png_encoder,'512','512','3','1536')] or body.count('SD_PNG_ENCODER ')!=1:
            raise RuntimeError('actual PNG encoder dispatch differs')
        entry['png_encoder_dispatch']=dict(mode=png_rows[0][0],width=512,height=512,channels=3,stride=1536)
        rgb_reused=int(args.rgb_buffer=='resident' and index>0)
        rgb_rows=re.findall(r'SD_RGB_BUFFER mode=(request|resident) allocated=(0|1) reused=(0|1) bytes=(\d+)',body)
        if rgb_rows!=[(args.rgb_buffer,str(1-rgb_reused),str(rgb_reused),'786432')] or body.count('SD_RGB_BUFFER ')!=1:
            raise RuntimeError('actual RGB buffer allocation or reuse differs')
        entry['rgb_buffer_dispatch']=dict(mode=args.rgb_buffer,allocated=1-rgb_reused,reused=rgb_reused,bytes=786432)
        pixel_rows=re.findall(r'SD_PIXEL_KERNEL mode=(generic|ve) clamp_calls=(\d+) pack_calls=(\d+) spatial=(\d+) bytes=(\d+)',body)
        if pixel_rows!=[(args.pixel_kernel,'1','1','262144','786432')] or body.count('SD_PIXEL_KERNEL ')!=1:
            raise RuntimeError('actual pixel kernel dispatch differs')
        entry['pixel_kernel_dispatch']=dict(mode=args.pixel_kernel,clamp_calls=1,pack_calls=1,spatial=262144,bytes=786432)
        output_rows=re.findall(r'SD_IMAGE_OUTPUT_PROFILE part=(\w+) seconds=([0-9.]+) calls=(\d+)',body)
        if args.image_output_profile:
            if [r[0] for r in output_rows]!=['prepare','pixel_clamp','decoded_trace','pixel_pack','png_encode'] or any(int(r[2])!=1 or not np.isfinite(float(r[1])) or float(r[1])<0 for r in output_rows):
                raise RuntimeError('complete finite image-output substage timing required')
        elif output_rows:
            raise RuntimeError('unexpected image-output profiling')
        entry['image_output_profile']=[dict(part=r[0],seconds=float(r[1]),calls=int(r[2])) for r in output_rows]
        operator_markers = re.findall(r'SD_OP_PROFILE stage=(clip|unet|vae) op=(\w+) seconds=([0-9.]+) nodes=(\d+)', body)
        if args.op_profile:
            if not all(any(row[0] == component for row in operator_markers) for component in ('clip','unet','vae')):
                raise RuntimeError('complete operator timing required')
            entry['operator_profile'] = [{'stage': stage, 'op': op, 'seconds': float(seconds), 'nodes': int(nodes)}
                                         for stage,op,seconds,nodes in operator_markers]
        elif operator_markers:
            raise RuntimeError('unexpected operator profiling')
        shape_lines = [line for line in body.splitlines() if line.startswith('SD_BINARY_SHAPE ')]
        if args.binary_shape_profile:
            for component in ('clip','unet','vae'):
                for op in ('ADD','MUL'):
                    expected=sum(int(n) for stage,name,_,n in operator_markers if stage==component and name==op)
                    if sum(line.startswith('SD_BINARY_SHAPE stage='+component+' op='+op+' ') for line in shape_lines)!=expected:
                        raise RuntimeError('binary shape count differs from operator nodes')
            entry['binary_shape_lines'] = shape_lines
        elif shape_lines:
            raise RuntimeError('unexpected binary shape profiling')
        generic_markers = re.findall(r'SD_GGML_THREADS stage=(clip|unet|vae) actual=(\d+) maximum=(\d+)', body)
        stage_markers=re.findall(r'SD_STAGE_THREADS stage=(clip|unet|vae) before=(\d+) configured=(\d+) after=(\d+) seconds=([0-9.]+)',body)
        if args.vae_threads==4:
            wanted_stages=['clip']+['unet']*cpu['steps']+['vae']
            if [r[0] for r in stage_markers]!=wanted_stages:
                raise RuntimeError('component stage thread markers differ')
            for component,before,configured,after,seconds in stage_markers:
                wanted=4 if component=='vae' else 8
                expected_before=4 if component=='clip' and index else 8
                if tuple(map(int,(before,configured,after)))!=(expected_before,wanted,wanted):
                    raise RuntimeError('component thread switch state differs')
        elif stage_markers:
            raise RuntimeError('fixed eight mode emitted stage switch markers')
        entry['stage_thread_switches']=[{'stage':a,'before':int(b),'configured':int(c),'after':int(d),'seconds':float(e)} for a,b,c,d,e in stage_markers]
        if unified:
            for component in ('clip','unet','vae'):
                markers=[row for row in generic_markers if row[0]==component]
                expected_calls=sum(int(calls) for stage,part,_,calls in profiles if stage==component and part=='backend_CPU')
                wanted=args.vae_threads if component=='vae' else 8
                if not markers or len(markers)!=expected_calls or any(tuple(map(int,row[1:]))!=(wanted,wanted) for row in markers):
                    raise RuntimeError('actual generic OpenMP region capacity differs')
        elif generic_markers:
            raise RuntimeError('pthread generic build emitted OpenMP region markers')
        if nlc_mode == 'openmp':
            for component in ('clip', 'unet', 'vae'):
                wanted = ((args.vae_blas_threads or args.vae_threads) if component=='vae' else 8) if unified else ({'clip': 1, 'unet': 8, 'vae': 4}[component] if args.nlc_threads == 'staged' else 1)
                markers = [row for row in thread_markers if row[0] == component]
                expected_calls = sum(int(calls) for stage, part, _, calls in profiles if stage == component and part == 'backend_BLAS')
                idle = 8 if unified and args.vae_blas_threads==4 else wanted if unified else 1
                if len(markers) != expected_calls or any(tuple(map(int, row[1:])) != (wanted, wanted, idle) for row in markers):
                    raise RuntimeError('actual graph thread capacity or idle reset differs')
        elif thread_markers:
            raise RuntimeError('sequential build emitted OpenMP thread markers')
        # Require actual NLC matrix assignment in every neural component/request.
        pending = 0
        assigned = set()
        for line in body.splitlines():
            if re.search(r'node\s+#\s*\d+\s+\(\s*MUL_MAT\):[^\[]*\[\s*BLAS\b', line):
                pending += 1
            marker = re.search(r'SD_NLC_GRAPH stage=(clip|unet|vae) nodes=\d+', line)
            if marker:
                if pending:
                    assigned.add(marker[1])
                pending = 0
        if assigned != {'clip', 'unet', 'vae'}:
            raise RuntimeError('actual NLC matrix assignment required in every request')
        entry.update(request_seconds=float(elapsed), weight_load_stages=[stage for _, stage in loads])
        entry['profile'] = [{'stage': stage, 'part': part, 'seconds': float(seconds), 'calls': int(calls)}
                            for stage, part, seconds, calls in profiles]
        entry['nlc_thread_graphs'] = {component: sum(row[0] == component for row in thread_markers) for component in ('clip', 'unet', 'vae')}
        entry['im2col_graphs'] = [{'stage': stage, 'optimized_nodes': int(nodes), 'fallback_nodes': int(fallback)}
                                 for stage, nodes, fallback in packing]
        source = reference/('case%d' % sequence[index])
        row = rows[sequence[index]]
        dest = folder/('request%d' % index)

        def array(name):
            return np.fromfile(source/(name+'.f32'), dtype=np.float32).reshape(row['arrays'][name]['shape'])

        def compare(name, expected, tolerance):
            actual = np.fromfile(dest/(name+'.f32'), dtype=np.float32)
            expected = np.asarray(expected, dtype=np.float32).reshape(-1)
            if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise RuntimeError('native output shape/nonfinite failure')
            delta = actual.astype(np.float64)-expected.astype(np.float64)
            relative = float(np.linalg.norm(delta)/max(float(np.linalg.norm(expected.astype(np.float64))), 1e-12))
            entry['checks'].append({'name': name, 'relative_l2': relative, 'max_abs_error': float(np.max(np.abs(delta))),
                                    'tolerance': tolerance, 'passed': relative <= tolerance})
            save()
            if not entry['checks'][-1]['passed']:
                raise RuntimeError('native request exceeded independent reference tolerance')

        if sha(dest/'native-noise.f32') != row['arrays']['noise']['sha256']:
            raise RuntimeError('request initial noise differs')
        compare('native-embeddings', array('embeddings'), 1e-4)
        for step in range(cpu['steps']):
            compare('native-step%d-input' % step, array('step%d-input' % step), 1e-4)
            compare('native-step%d-epsilon' % step, array('step%d-epsilon' % step), 1e-3)
            timestep = np.fromfile(dest/('native-step%d-timestep.f32' % step), dtype=np.float32)
            if timestep.size != 1 or abs(float(timestep[0])-float(array('timesteps').reshape(-1)[step])) > 1e-3:
                raise RuntimeError('request timestep differs')
        compare('native-latent', array('step%d-latent' % (cpu['steps']-1)), 1e-2)
        compare('native-decoded', np.clip((array('decoded')+1)*.5, 0, 1), 1e-2)
        if not (dest/'image.png').is_file():
            raise RuntimeError('request image missing')
        entry['trace_sha256'] = {path.name: sha(path) for path in sorted(dest.glob('*.f32'))}
        entry['png_sha256'] = sha(dest/'image.png')
        save()
        print('Request', index, args.mode, 'seconds', elapsed, 'loads', len(loads), 'checks', len(entry['checks']), 'PASS', flush=True)
    report['completed'] = True
    save()
    print('Native sequential requests verified:', folder.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
