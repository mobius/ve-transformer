"""Paired full-model experiment; invoke through the temperature guard."""
import json
import statistics
import subprocess
import time
from pathlib import Path
from check_qwen_model import ROOT, MODEL, ENV, PROMPTS, chat, verify_model

verify_model()
folder = ROOT / 'build/results' / time.strftime('%Y%m%dT%H%M%SZ-qwen-block-scales', time.gmtime())
folder.mkdir(parents=True)
results = {}
for label, binary in [('previous', 'qwen-infer-ve-nlc-pool-old'),
                      ('block-scales', 'qwen-infer-ve-nlc'),
                      ('direct-quant', 'qwen-infer-ve')]:
    command = ['ve_exec', '-N', '1', str(ROOT/'build'/binary), '--model', str(MODEL),
               '--prompt', chat(PROMPTS[0]), '--tokens', '4', '--threads', '8',
               '--context', '4096', '--repeats', '3', '--force-count', '--no-mmap']
    print('Running '+label, flush=True)
    with (folder/(label+'.stderr.log')).open('w') as err:
        completed = subprocess.run(command, env=ENV, stdout=subprocess.PIPE, stderr=err,
                                   text=True, timeout=1800, check=True)
    (folder/(label+'.jsonl')).write_text(completed.stdout)
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(records) != 3 or any(row['generated_tokens'] != 4 for row in records):
        raise RuntimeError('incomplete paired experiment')
    for row in records:
        if row['model_parameters'] != 34660610688 or row['tokens'] != [18,24,16,248046]:
            raise RuntimeError('unexpected model or arithmetic output')
        print(json.dumps(dict(label=label, **row)), flush=True)
    results[label] = dict(steady_decode_tokens_per_second=statistics.median(
        row['decode_tokens_per_second'] for row in records[1:]),
        steady_prefill_seconds=statistics.median(row['prefill_seconds'] for row in records[1:]))
results['block_scales_speedup'] = (results['block-scales']['steady_decode_tokens_per_second'] /
                                  results['previous']['steady_decode_tokens_per_second'])
(folder/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results), flush=True)
print('artifacts='+str(folder.relative_to(ROOT)), flush=True)
