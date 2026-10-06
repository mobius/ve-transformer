"""Real cached execution against the preserved, accepted FP64 CPU references."""
import json
from pathlib import Path
import shutil
import statistics
import time
from check_qwen_model import ROOT, PROMPTS, chat, verify_model, run, compare


def main():
    reference = ROOT/'build/results/20261002T151124Z-qwen36'
    accepted = json.loads((reference/'numerical-progress.json').read_text())
    if accepted['status']!='correctness_complete' or not accepted['cache_reset_completed']:
        raise RuntimeError('reference suite has not passed')
    verify_model()
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen-dense-cache',time.gmtime())
    folder.mkdir(exist_ok=False)
    shutil.copyfile(reference/'build-manifest.json',folder/'reference-build-manifest.json')
    reports=[]
    for index,prompt in enumerate(PROMPTS):
        cpu='case{}-cpu'.format(index);cached='case{}-cached'.format(index)
        for suffix in ('-0.f32','.jsonl','.tokens'):
            shutil.copyfile(reference/(cpu+suffix),folder/(cpu+suffix))
        ref=json.loads((reference/(cpu+'.jsonl')).read_text())
        actual=run(folder,cached,'accurate',chat(prompt),count=12,
                   forced=folder/(cpu+'.tokens'),trace=True,dense_cache_mib=16384)[0]
        reports.append(compare(folder,cpu,cached,ref,actual,12))
    reset=run(folder,'reset-cached','accurate',chat(PROMPTS[1]),count=16,
              repeats=3,dense_cache_mib=16384)
    baseline=[json.loads(line) for line in (reference/'reset-ve.jsonl').read_text().splitlines()]
    if any(record['tokens']!=baseline[0]['tokens'] for record in reset):
        raise RuntimeError('cached reset output differs from accepted uncached output')
    base_rate=statistics.median(row['decode_tokens_per_second'] for row in baseline[1:])
    cached_rate=statistics.median(row['decode_tokens_per_second'] for row in reset[1:])
    report=dict(full_vocabulary_cases=reports,cache_reset_repeats=3,
                uncached_tokens_per_second=base_rate,cached_tokens_per_second=cached_rate,
                speedup=cached_rate/base_rate,performance_scan_completed=False)
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Dense cache validation completed; artifacts='+str(folder.relative_to(ROOT)),flush=True)


if __name__=='__main__':
    main()
