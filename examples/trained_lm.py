"""Greedy generation using a fixed, trained TinyStories checkpoint."""
from pathlib import Path
import argparse
import json
import sys
import time
from contextlib import ExitStack
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from gpt_neo_ve import GPTNeoNative, load_model, greedy_generate
from resident_ve import GPTNeoResident

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['torch', 'cpu', 'fast', 'resident'], default='resident')
    parser.add_argument('--threads', type=int, choices=[1,2,4,8], default=1)
    parser.add_argument('--node', type=int, default=1)
    parser.add_argument('--new-tokens', type=int, default=16)
    parser.add_argument('--prompt', default='Once upon a time there was a little girl')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, tokenizer = load_model()
    ids = tokenizer.encode(args.prompt, return_tensors='pt')[0]
    if len(ids)+args.new_tokens > 256:
        parser.error('initial demonstration limits total length to 256 tokens')
    with ExitStack() as resources:
        initialization = time.perf_counter()
        if args.backend == 'resident':
            native = resources.enter_context(GPTNeoResident(
                model, args.node, 'fast', capacity=256, threads=args.threads))
        elif args.backend != 'torch':
            native = GPTNeoNative(model, args.node, args.backend)
        else:
            native = None
        initialization_ms = (time.perf_counter()-initialization)*1000
        def forward_last(tokens):
            if native:
                return native.logits(tokens, last_only=True)[0]
            hidden = model.transformer(tokens.unsqueeze(0), use_cache=False).last_hidden_state[0, -1:]
            return model.lm_head(hidden)[0]
        with torch.inference_mode():
            forward_last(ids)  # Warm up before measuring generation.
            start = time.perf_counter()
            output, generated = greedy_generate(forward_last, ids, args.new_tokens,
                                                 model.config.eos_token_id)
            elapsed = time.perf_counter()-start
        print(json.dumps(dict(backend=args.backend, node=args.node,
                             ve_threads=args.threads if args.backend == 'resident' else None,
                             native_initialization_ms=initialization_ms,
                             prompt_tokens=len(ids), generated_tokens=len(generated),
                             generated_ids=generated, elapsed_seconds=elapsed,
                             generated_tokens_per_second=len(generated)/elapsed,
                             text=tokenizer.decode(output, skip_special_tokens=True)),
                         ensure_ascii=False, indent=2))
