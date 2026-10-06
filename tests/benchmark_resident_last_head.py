"""Match CPU and resident VE output-head work: only the last position, no cache."""
from pathlib import Path
import json
import argparse
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from gpt_neo_ve import load_model, greedy_generate
from resident_ve import GPTNeoResident

parser=argparse.ArgumentParser()
parser.add_argument('--node',type=int,default=1)
args=parser.parse_args()
if args.node<0: parser.error('node must be nonnegative')
torch.set_num_threads(1)
model, tokenizer = load_model()
ids = tokenizer.encode('Once upon a time there was a little girl',return_tensors='pt')[0]
with torch.inference_mode():
    def cpu_last(tokens):
        hidden = model.transformer(tokens.unsqueeze(0),use_cache=False).last_hidden_state[0,-1:]
        return model.lm_head(hidden)[0]
    expected = model(ids.unsqueeze(0),use_cache=False).logits[0,-1]
    torch.testing.assert_close(cpu_last(ids), expected, atol=2e-4,rtol=2e-4)
    ref, ref_ids = greedy_generate(cpu_last,ids,16,model.config.eos_token_id)
    with GPTNeoResident(model,args.node,'fast',capacity=512,threads=1) as native:
        def ve_last(tokens):
            return native.logits(tokens,last_only=True)[0]
        cpu_last(ids); ve_last(ids)
        samples={'torch-last-head':[], 'resident-last-head':[]}
        for repeat in range(5):
            # Alternate order to reduce warm-up/order bias.
            modes=['torch-last-head','resident-last-head']
            if repeat%2: modes.reverse()
            for mode in modes:
                start=time.perf_counter()
                output, generated=greedy_generate(cpu_last if mode=='torch-last-head' else ve_last,
                                                  ids,16,model.config.eos_token_id)
                elapsed=time.perf_counter()-start
                assert generated==ref_ids and torch.equal(output,ref)
                samples[mode].append(elapsed)
        print(json.dumps(dict(samples_s=samples,
            mean_tokens_per_second={k:16/statistics.mean(v) for k,v in samples.items()},
            ids_match=True,ve_threads=1,cpu_threads=1,cache=False,output_head='last position')))
