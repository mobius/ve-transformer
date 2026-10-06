"""Real 35B-A3B comparison. Must be launched through validate_qwen.sh thermal guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'build/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
SHA256 = 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61'
PROMPTS = ['Compute 17 * 23. Give only the numeric answer.',
           'Write a Python function that adds two integers.',
           '请用一句话说明二分查找的原理。']
ENV = os.environ.copy()
ENV.update(VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib',
           OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
CPU_BACKENDS=('cpu','cpu_float','cpu_accurate')


def chat(text):
    # Official template enable_thinking=false, text-only user message.
    return '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def verify_model():
    if MODEL.stat().st_size != 22134528992:
        raise RuntimeError('incorrect model size')
    digest = hashlib.sha256()
    with MODEL.open('rb') as stream:
        for chunk in iter(lambda: stream.read(32*1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError('model checksum mismatch')


def pin_accurate_build(folder, binary):
    pinned = folder / 'build-manifest.json'
    if not pinned.exists():
        source = ROOT / 'build/qwen-ve-accurate/build-manifest.json'
        pinned.write_bytes(source.read_bytes())
    manifest = json.loads(pinned.read_text())
    relative = str(binary.relative_to(ROOT))
    expected = next(row for row in manifest['artifacts'] if row['path'] == relative)
    digest = hashlib.sha256()
    with binary.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected['sha256']:
        raise RuntimeError('executor differs from pinned build: ' + relative)


def run(folder, label, backend, prompt, threads=8, count=12, repeats=1, forced=None, trace=False, binary_override=None, timeout_seconds=1800, dense_cache_mib=0):
    binaries = {'cpu': ROOT/'build/qwen-cpu/qwen-infer',
                'cpu_float': ROOT/'build/qwen-cpu/qwen-infer-cpu-float',
                'cpu_accurate': ROOT/'build/qwen-cpu/qwen-infer-cpu-accurate',
                'accurate': ROOT/'build/qwen-ve-accurate/qwen-infer-ve',
                'nlc': ROOT/'build/qwen-infer-ve-nlc',
                'baseline': ROOT/'build/qwen-infer-ve-baseline',
                've': ROOT/'build/qwen-infer-ve'}
    if binary_override is not None:binaries[backend]=binary_override
    if backend in ('cpu_accurate', 'accurate') and binary_override is None:
        pin_accurate_build(folder, binaries[backend])
    cmd = ['/opt/nec/ve/bin/ve_exec', '-N', '1'] if backend not in CPU_BACKENDS else ['taskset','-c','0-23']
    cmd += [str(binaries[backend]), '--model', str(MODEL), '--prompt', prompt,
            '--threads', str(threads), '--tokens', str(count), '--context', '4096',
            '--repeats', str(repeats), '--force-count']
    if backend not in CPU_BACKENDS: cmd += ['--no-mmap']
    if dense_cache_mib:
        if backend not in ('cpu_accurate', 'accurate'):
            raise ValueError('dense cache requires the accurate backend')
        cmd += ['--dense-cache-mib',str(dense_cache_mib)]
    if trace: cmd += ['--trace', str(folder/label)]
    if forced: cmd += ['--forced-tokens', str(forced)]
    print('Running {}: backend={} threads={} generated={}'.format(label,backend,threads,count), flush=True)
    samples=[]
    stopped=threading.Event()
    def observe_memory(pid):
        while not stopped.is_set():
            try:
                probe=subprocess.run(['ve-ps','-p',str(pid),'-o','pid=,rss='],
                                     stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                     text=True,timeout=2)
                for line in probe.stdout.splitlines():
                    fields=line.split()
                    if len(fields)==2 and fields[0]==str(pid):
                        samples.append(dict(monotonic_seconds=time.monotonic(),rss_raw=int(fields[1])))
            except (ValueError,subprocess.TimeoutExpired):
                pass  # A process may exit while ve-ps is fetching its statistics.
            stopped.wait(5)
    result_path = folder/(label+'.jsonl')
    with (folder/(label+'.stderr.log')).open('w') as err, result_path.open('w') as output_stream:
        process=subprocess.Popen(cmd,env=ENV,cwd=str(ROOT),stdout=output_stream,
                                 stderr=err,text=True)
        observer=None
        if backend not in CPU_BACKENDS:
            observer=threading.Thread(target=observe_memory,args=(process.pid,),daemon=True)
            observer.start()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()
            raise
        finally:
            stopped.set()
            if observer:observer.join(timeout=3)
            if observer:
                (folder/(label+'.memory.json')).write_text(json.dumps(dict(
                    field='ve-ps RSS',peak_raw=max((s['rss_raw'] for s in samples),default=None),
                    sample_interval_seconds=5,samples=samples),indent=2)+'\n')
    output = result_path.read_text()
    if process.returncode:
        raise RuntimeError('inference failed: {} exit {}, see local stderr log'.format(label,process.returncode))
    records = [json.loads(line) for line in output.splitlines()]
    if len(records)!=repeats:
        raise RuntimeError('missing result records')
    for record in records:
        if record['generated_tokens']!=count or record['model_parameters']<34_000_000_000:
            raise RuntimeError('wrong model or incomplete generation')
        if record['vocab']!=248320:
            raise RuntimeError('unexpected vocabulary')
        if backend in ('cpu_accurate','accurate') and record.get('math_mode')!='fp64_accumulation':
            raise RuntimeError('wrong accumulation mode')
        if dense_cache_mib:
            if record.get('dense_cache_budget_bytes') != dense_cache_mib*1024**2:
                raise RuntimeError('wrong dense cache budget')
            if record['dense_cache_retained_bytes']>record['dense_cache_budget_bytes']:
                raise RuntimeError('dense cache exceeded payload budget')
            if record.get('dense_cache_reserved_bytes',0):
                raise RuntimeError('cache fill reservation remained after inference')
            if not record['dense_cache_hits'] or not record['dense_cache_entries']:
                raise RuntimeError('dense cache was not exercised')
        print(json.dumps(dict(label=label, **record), ensure_ascii=False), flush=True)
    return records


def compare(folder, cpu_label, ve_label, reference, actual, count):
    ref=np.fromfile(folder/(cpu_label+'-0.f32'),dtype='<f4').reshape(count,reference['vocab'])
    got=np.fromfile(folder/(ve_label+'-0.f32'),dtype='<f4').reshape(count,actual['vocab'])
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise RuntimeError('nonfinite trace')
    diff=got.astype(np.float64)-ref
    maximum=float(np.max(np.abs(diff)));rmse=float(np.sqrt(np.mean(diff**2)))
    agreement=int(np.sum(np.argmax(ref,axis=1)==np.argmax(got,axis=1)))
    report=dict(reference=cpu_label,actual=ve_label,max_abs_error=maximum,rmse=rmse,
                argmax_agreement=agreement,steps=count)
    print('LOGIT_COMPARISON '+json.dumps(report),flush=True)
    # Predetermined numerical gate: never relaxed automatically to make a run pass.
    if maximum>0.1 or rmse>0.01 or agreement!=count:
        raise RuntimeError('full-vocabulary numerical or greedy agreement check failed')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','correctness','bench','full'],default='full')
    parser.add_argument('--backend',choices=['quant','nlc','accurate'],default='quant')
    args=parser.parse_args()
    cpu_backend={'quant':'cpu','nlc':'cpu_float','accurate':'cpu_accurate'}[args.backend]
    ve_backend={'quant':'ve','nlc':'nlc','accurate':'accurate'}[args.backend]
    verify_model()
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen36',time.gmtime())
    folder.mkdir(parents=True,exist_ok=False)
    reports=[]
    if args.mode=='smoke':
        for backend in (ve_backend,cpu_backend,'baseline'):
            run(folder,'smoke-'+backend,backend,chat(PROMPTS[0]),threads=1 if backend in CPU_BACKENDS else 8,count=4)
    if args.mode in ('correctness','full'):
        for i,prompt in enumerate(PROMPTS):
            count=12;cpu='case{}-cpu'.format(i);ve='case{}-ve'.format(i)
            ref=run(folder,cpu,cpu_backend,chat(prompt),threads=1,count=count,trace=True)[0]
            forced=folder/(cpu+'.tokens');forced.write_text(' '.join(map(str,ref['tokens'])))
            out=run(folder,ve,ve_backend,chat(prompt),count=count,forced=forced,trace=True)[0]
            reports.append(compare(folder,cpu,ve,ref,out,count))
        # Repeated requests exercise actual cache reset rather than independent workers.
        reset=run(folder,'reset-ve',ve_backend,chat(PROMPTS[1]),count=16,repeats=3)
        if any(r['tokens']!=reset[0]['tokens'] for r in reset[1:]):
            raise RuntimeError('cache reset changed greedy output')
    if args.mode in ('bench','full'):
        base=run(folder,'baseline-t8','baseline',chat(PROMPTS[1]),count=32,repeats=3,timeout_seconds=7200)
        medbase=statistics.median(r['decode_tokens_per_second'] for r in base[1:])
        for threads in (1,2,4,8):
            records=run(folder,'optimized-t'+str(threads),ve_backend,chat(PROMPTS[1]),
                        threads=threads,count=32,repeats=3,timeout_seconds=7200)
            median=statistics.median(r['decode_tokens_per_second'] for r in records[1:])
            reports.append(dict(threads=threads,decode_tokens_per_second=median,speedup_over_baseline=median/medbase))
        # Separate longer-prompt prefill and decode; no extrapolation to 256K.
        for paragraphs in (16,64,128):
            prompt=chat(('A sorted list allows binary search to reduce the remaining interval by half.\n'*paragraphs)+
                        'Explain why binary search is efficient.')
            run(folder,'long-{}'.format(paragraphs),ve_backend,prompt,count=16,repeats=2,timeout_seconds=7200)
    (folder/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Real model validation completed; artifacts='+str(folder.relative_to(ROOT)),flush=True)


if __name__=='__main__':
    main()
