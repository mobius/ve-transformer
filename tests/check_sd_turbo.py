"""Guarded SD-Turbo VE comparison with an independent exported CPU reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import re
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from export_sd_reference import PROMPTS

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':raise RuntimeError('temperature supervision required')
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--framework',choices=('compact','modern'),default='compact')
    p.add_argument('--recheck',type=Path,help='reanalyze a completed native process without running hardware again')
    p.add_argument('--threads',type=int,choices=(1,2,4,8),default=8)
    p.add_argument('--threadpool',choices=('disposable','persistent'),default='disposable')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--op-profile',action='store_true')
    p.add_argument('--silu',choices=('generic','ve'),default='generic')
    p.add_argument('--softmax',choices=('generic','ve'),default='generic')
    p.add_argument('--im2col',choices=('generic','ve'),default='generic')
    args=p.parse_args()
    if args.op_profile and not args.profile:
        raise RuntimeError('operator profiling requires stage profiling')
    if args.framework!='compact' and (args.profile or args.op_profile or args.silu!='generic' or args.softmax!='generic' or args.im2col!='generic' or args.threadpool!='disposable'):
        raise RuntimeError('profiling and pool experiment require the compact VE build')
    reference=args.reference.resolve();reference.relative_to(ROOT/'build')
    cpu=json.loads((reference/'summary.json').read_text())
    if not cpu['completed'] or cpu['size']!=512:raise RuntimeError('complete 512 CPU reference required')
    binary=ROOT/('build/sd-baseline-ve/bin/sd' if args.framework=='compact' else 'build/sd-ve/bin/sd-cli')
    manifest=json.loads((binary.parent.parent/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('image executable/source checksum mismatch')
    model=ROOT/'build/models/sd-turbo'
    provenance=json.loads((model/'provenance.json').read_text())
    if not provenance['completed'] or cpu['model_revision']!=provenance['revision']:raise RuntimeError('model provenance mismatch')
    for name,entry in provenance['files'].items():
        if sha(model/name)!=entry['sha256']:raise RuntimeError('official weight checksum mismatch')
    folder=args.recheck.resolve() if args.recheck else ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-ve',time.gmtime())
    folder.relative_to(ROOT/'build')
    if not args.recheck:folder.mkdir()
    report={'completed':False,'adopted':False,'runtime_slot':1,'precision':'FP32 weights and convolution input packing',
            'framework_variant':args.framework,
            'size':512,'steps':cpu['steps'],'threads':args.threads,'binary_sha256':sha(binary),
            'threadpool_mode':args.threadpool,'profile_enabled':args.profile,
            'operator_profile_enabled':args.op_profile,'silu_mode':args.silu,'softmax_mode':args.softmax,'im2col_mode':args.im2col,
            'model_revision':provenance['revision'],'test_sha256':sha(Path(__file__)),
            'reference_artifacts':str(reference.relative_to(ROOT)),
            'source_scope':'independent Diffusers CPU comparison; saved noise and custom sigma schedule',
            'cases':[]}
    if args.recheck:
        report=json.loads((folder/'summary.json').read_text())
        if (report['binary_sha256']!=sha(binary) or report['reference_artifacts']!=str(reference.relative_to(ROOT))
                or report['model_revision']!=provenance['revision'] or report['steps']!=cpu['steps']
                or report['framework_variant']!=args.framework or report['threads']!=args.threads):
            raise RuntimeError('recheck provenance mismatch')
        if report.get('threadpool_mode','disposable')!=args.threadpool or report.get('profile_enabled',False)!=args.profile:
            raise RuntimeError('recheck execution settings differ')
        if report.get('im2col_mode','generic')!=args.im2col:
            raise RuntimeError('recheck im2col mode differs')
        if report.get('softmax_mode','generic')!=args.softmax:
            raise RuntimeError('recheck Softmax mode differs')
        if report.get('silu_mode','generic')!=args.silu:
            raise RuntimeError('recheck SiLU mode differs')
        if report.get('operator_profile_enabled',False)!=args.op_profile:
            raise RuntimeError('recheck operator profiling differs')
        if sha(folder/'check_sd_turbo.py')!=report['test_sha256']:
            raise RuntimeError('original execution checker snapshot differs')
        report['completed']=False;report['recheck_sha256']=sha(Path(__file__))
        (folder/'recheck_sd_turbo.py').write_bytes(Path(__file__).read_bytes())
    def save():(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    if not args.recheck:(folder/'check_sd_turbo.py').write_bytes(Path(__file__).read_bytes())
    for row in cpu['cases']:
        index=row['case'];src=reference/('case%d'%index);dest=folder/('case%d'%index)
        if not args.recheck:dest.mkdir()
        for name,entry in row['arrays'].items():
            if sha(src/(name+'.f32'))!=entry['sha256']:raise RuntimeError('reference array checksum mismatch')
        def reference_array(name):
            metadata=row['arrays'][name]
            value=np.fromfile(src/(name+'.f32'),dtype=np.float32)
            return value.reshape(metadata['shape'])
        sigma=reference_array('sigmas').reshape(-1)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib',
                 SD_FIXED_NOISE=str(src/'noise.f32'),SD_TRACE_DIR=str(dest))
        command=['ve_exec','-N','1',str(binary),'-m',str(model),'-p',PROMPTS[index],
                 '-W','512','-H','512','--steps',str(cpu['steps']),'--cfg-scale','1',
                 '--clip-skip','2',
                 '--prediction','eps','--sampling-method','euler','--scheduler','discrete',
                 '--sigmas',','.join(format(float(v),'.9g') for v in sigma),
                 '--type','f32','--backend','BLAS','--params-backend','CPU','--auto-fit','off',
                 '-t',str(args.threads),'-s',str(row['seed']),'-o',str(dest/'image.png')]
        if args.framework=='compact':
            env.update(SD_TURBO_EPS='1',SD_FIXED_SIGMAS=str(src/'sigmas.f32'),GGML_SCHED_DEBUG='2',
                       SD_PROFILE='1' if args.profile else '0',
                       SD_OP_PROFILE='1' if args.op_profile else '0',
                       SD_VE_SILU='1' if args.silu=='ve' else '0',
                       SD_VE_SOFTMAX='1' if args.softmax=='ve' else '0',
                       SD_VE_IM2COL='1' if args.im2col=='ve' else '0',
                       SD_PERSISTENT_THREADPOOL='1' if args.threadpool=='persistent' else '0')
            command=['ve_exec','-N','1',str(binary),'-m',str(model),'-p',PROMPTS[index],
                     '-W','512','-H','512','--steps',str(cpu['steps']),'--cfg-scale','1',
                     '--clip-skip','2','--sampling-method','euler','--schedule','discrete',
                     '--type','f32','-t',str(args.threads),'-s',str(row['seed']),'-o',str(dest/'image.png')]
        if args.recheck:
            print('Rechecking completed native case',index,flush=True)
            entry=next((entry for entry in report['cases'] if entry['case']==index),None)
            if entry is None or entry['returncode']!=0 or not (dest/'image.png').exists():
                raise RuntimeError('successful native process and image required for recheck')
            entry['checks']=[]
        else:
            print('Native SD-Turbo case',index,'steps',cpu['steps'],flush=True)
            started=time.perf_counter()
            with (dest/'native.log').open('w') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            entry={'case':index,'process_seconds':time.perf_counter()-started,'returncode':result.returncode,'checks':[]}
            report['cases'].append(entry);save()
            if result.returncode:raise RuntimeError('native image inference failed; inspect private log')
        if args.framework=='compact':
            native_log=(dest/'native.log').read_text(errors='replace')
            if args.profile:
                entries=re.findall(r'SD_PROFILE stage=([a-z]+) part=([a-zA-Z_]+) seconds=([0-9.]+) calls=(\d+)',native_log)
                entry['profile']=[{'stage':stage,'part':part,'seconds':float(seconds),'calls':int(calls)}
                                  for stage,part,seconds,calls in entries]
                if not all(any(e['stage']==stage and e['part']==part for e in entry['profile'])
                           for stage in ('clip','unet','vae')
                           for part in ('weights_load','graph_build','graph_alloc','graph_compute','backend_CPU','backend_BLAS')):
                    raise RuntimeError('complete stage profiling required')
                if not all(np.isfinite(e['seconds']) and e['seconds']>=0 for e in entry['profile']):
                    raise RuntimeError('invalid profiling duration')
            packing=re.findall(r'SD_IM2COL stage=(clip|unet|vae) optimized_nodes=(\d+) fallback_nodes=(\d+)',native_log)
            entry['im2col_graphs']=[{'stage':stage,'optimized_nodes':int(nodes),'fallback_nodes':int(fallback)} for stage,nodes,fallback in packing]
            if args.im2col=='ve' and (not all(any(e['stage']==stage and e['optimized_nodes']>0 for e in entry['im2col_graphs']) for stage in ('unet','vae'))
                                    or any(e['fallback_nodes'] for e in entry['im2col_graphs'])):
                raise RuntimeError('actual im2col optimization required in UNet/VAE without fallback')
            if args.im2col=='generic' and packing:raise RuntimeError('generic unexpectedly activated im2col candidate')
            if args.op_profile:
                items=re.findall(r'SD_OP_PROFILE stage=(clip|unet|vae) op=([A-Z0-9_]+) seconds=([0-9.]+) nodes=(\d+)',native_log)
                entry['operator_profile']=[{'stage':stage,'op':op,'seconds':float(seconds),'nodes':int(nodes)}
                                           for stage,op,seconds,nodes in items]
                if not all(any(e['stage']==stage for e in entry['operator_profile']) for stage in ('clip','unet','vae')):
                    raise RuntimeError('three operator profiles required')
                if not all(np.isfinite(e['seconds']) and e['seconds']>=0 and e['nodes']>0
                           for e in entry['operator_profile']):
                    raise RuntimeError('invalid operator timing')
            pools=re.findall(r'SD_THREADPOOL stage=(clip|unet|vae) threads=(\d+) poll=0',native_log)
            entry['persistent_pool_stages']=[{'stage':stage,'threads':int(threads)} for stage,threads in pools]
            if args.threadpool=='persistent' and (len(pools)!=3 or
                    {stage for stage,threads in pools}!={'clip','unet','vae'} or
                    any(int(threads)!=args.threads for stage,threads in pools)):
                raise RuntimeError('three correctly sized component pools required')
            if args.threadpool=='disposable' and pools:
                raise RuntimeError('baseline unexpectedly attached a persistent pool')
            assignments=re.findall(r'SD_NLC_GRAPH stage=(clip|unet|vae) nodes=(\d+)',native_log)
            entry['nlc_graphs']=[{'stage':name,'nodes':int(count)} for name,count in assignments]
            save()
            if not all(any(name==stage and int(count)>0 for name,count in assignments)
                       for stage in ('clip','unet','vae')):
                raise RuntimeError('all neural stages must demonstrate NLC graph assignment')
            matmuls=[];pending=0
            for line in native_log.splitlines():
                if re.search(r'node\s+#\s*\d+\s+\(\s*MUL_MAT\):[^\[]*\[\s*BLAS\b',line):
                    pending+=1
                marker=re.search(r'SD_NLC_GRAPH stage=(clip|unet|vae) nodes=\d+',line)
                if marker:
                    matmuls.append({'stage':marker[1],'matmul_nodes':pending});pending=0
            entry['nlc_matmul_detail_collected']=any(row['matmul_nodes']>0 for row in matmuls)
            if entry['nlc_matmul_detail_collected']:
                entry['nlc_matmul_graphs']=matmuls
            if not args.recheck and not all(any(row['stage']==stage and row['matmul_nodes']>0 for row in matmuls)
                                           for stage in ('clip','unet','vae')):
                raise RuntimeError('all neural stages must demonstrate NLC matrix assignment')
            save()
        if sha(dest/'native-noise.f32')!=row['arrays']['noise']['sha256']:raise RuntimeError('initial noise differs')
        def compare(native,expected,tolerance):
            actual=np.fromfile(dest/(native+'.f32'),dtype=np.float32)
            expected=np.asarray(expected,dtype=np.float32).reshape(-1)
            if actual.shape!=expected.shape or not np.isfinite(actual).all():raise RuntimeError('native output shape/nonfinite failure')
            delta=actual.astype(np.float64)-expected.astype(np.float64)
            relative=float(np.sqrt(np.sum(delta*delta))/max(float(np.sqrt(np.sum(expected.astype(np.float64)**2))),1e-12))
            maximum=float(np.max(np.abs(delta)))
            check={'name':native,'relative_l2':relative,'max_abs_error':maximum,'tolerance':tolerance,'passed':relative<=tolerance}
            entry['checks'].append(check);save()
            print(native,'relative_l2',relative,'max_abs',maximum,'PASS' if check['passed'] else 'FAIL',flush=True)
            if not check['passed']:raise RuntimeError('native stage exceeded planned numerical tolerance')
        compare('native-embeddings',reference_array('embeddings'),1e-4)
        for step in range(cpu['steps']):
            compare('native-step%d-input'%step,reference_array('step%d-input'%step),1e-4)
            compare('native-step%d-epsilon'%step,reference_array('step%d-epsilon'%step),1e-3)
            t=np.fromfile(dest/('native-step%d-timestep.f32'%step),dtype=np.float32)
            if t.size!=1 or abs(float(t[0])-float(reference_array('timesteps').reshape(-1)[step]))>1e-3:
                raise RuntimeError('native timestep differs')
        compare('native-latent',reference_array('step%d-latent'%(cpu['steps']-1)),1e-2)
        # Native VAE returns [0,1] pixels, while the independent raw decode is [-1,1].
        compare('native-decoded',np.clip((reference_array('decoded')+1)*.5,0,1),1e-2)
        save()
    report['completed']=True;save();print('Native full image stages match CPU reference:',folder.relative_to(ROOT),flush=True)

if __name__=='__main__':main()
