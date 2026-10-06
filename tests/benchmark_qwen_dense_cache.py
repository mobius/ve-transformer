"""Final same-math baseline, thread sweep and long-input measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time
from check_qwen_model import ROOT, PROMPTS, chat, verify_model, run, pin_accurate_build


def measure(records):
    warm=records[1:]
    return dict(prompt_tokens=records[0]['prompt_tokens'],
                cold_prefill_seconds=records[0]['prefill_seconds'],
                warm_prefill_seconds=statistics.median(row['prefill_seconds'] for row in warm),
                warm_decode_tokens_per_second=statistics.median(row['decode_tokens_per_second'] for row in warm))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--accepted',type=Path,required=True)
    args=parser.parse_args()
    accepted=args.accepted.resolve()
    gate=json.loads((accepted/'summary.json').read_text())
    cases=gate['full_vocabulary_cases']
    if len(cases)!=3 or gate['cache_reset_repeats']!=3 or any(
            row['steps']!=12 or row['argmax_agreement']!=12 or
            row['max_abs_error']>0.1 or row['rmse']>0.01 for row in cases):
        raise RuntimeError('complete cache correctness acceptance required')
    # Tie the benchmark to the exact candidate that passed acceptance.
    pin_accurate_build(accepted,ROOT/'build/qwen-ve-accurate/qwen-infer-ve')
    baseline=ROOT/'build/qwen-accurate-baseline/qwen-infer-ve'
    manifest=json.loads((baseline.parent/'build-manifest.json').read_text())
    expected=next(row['sha256'] for row in manifest['artifacts']
                  if row['path']=='build/qwen-ve-accurate/qwen-infer-ve')
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('preserved uncached executor changed')
    verify_model()
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen-final-bench',time.gmtime())
    folder.mkdir(exist_ok=False)
    shutil.copyfile(baseline.parent/'build-manifest.json',folder/'baseline-build-manifest.json')
    base=run(folder,'uncached-t8','accurate',chat(PROMPTS[1]),threads=8,
             count=32,repeats=3,binary_override=baseline,timeout_seconds=7200)
    report=dict(accepted=str(accepted.relative_to(ROOT)),baseline=measure(base),threads={},long_inputs={})
    for threads in (1,2,4,8):
        rows=run(folder,'cached-t'+str(threads),'accurate',chat(PROMPTS[1]),
                 threads=threads,count=32,repeats=3,timeout_seconds=7200,dense_cache_mib=16384)
        if any(row['tokens']!=base[0]['tokens'] for row in rows):
            raise RuntimeError('thread/cache configuration changed the baseline greedy sequence')
        result=measure(rows)
        result['speedup']=result['warm_decode_tokens_per_second']/report['baseline']['warm_decode_tokens_per_second']
        report['threads'][str(threads)]=result
        (folder/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    selected=max(report['threads'],key=lambda key:report['threads'][key]['warm_decode_tokens_per_second'])
    report['selected_threads']=int(selected)
    for paragraphs in (16,64,128):
        prompt=chat(('A sorted list allows binary search to reduce the remaining interval by half.\n'*paragraphs)+
                    'Explain why binary search is efficient.')
        rows=run(folder,'long-'+str(paragraphs),'accurate',prompt,threads=int(selected),
                 count=16,repeats=2,timeout_seconds=7200,dense_cache_mib=16384)
        if rows[0]['tokens']!=rows[1]['tokens']:
            raise RuntimeError('long-input reset changed output')
        report['long_inputs'][str(paragraphs)]=measure(rows)
        (folder/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['performance_scan_completed']=True
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Final performance scan completed; artifacts='+str(folder.relative_to(ROOT)),flush=True)


if __name__=='__main__':
    main()
