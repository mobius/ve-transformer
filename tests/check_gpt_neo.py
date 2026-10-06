"""Check decoder semantics and actual author-trained generation on CPU/VE."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from gpt_neo_ve import GPTNeoNative, infer_block, load_model, greedy_generate


def check_random(devices):
    torch.manual_seed(1102)
    for activation in ('gelu_new', 'relu'):
        config = GPTNeoConfig(vocab_size=32, hidden_size=16, num_layers=2,
                              num_heads=4, intermediate_size=31, max_position_embeddings=16,
                              attention_types=[[['global', 'local'], 1]], window_size=3,
                              activation_function=activation, layer_norm_epsilon=1e-4,
                              resid_dropout=0, embed_dropout=0, attention_dropout=0)
        config._attn_implementation = 'eager'
        model = GPTNeoForCausalLM(config).eval()
        for block in model.transformer.h:
            # Nonzero projections and norms exercise all six optional biases.
            for layer in (block.attn.attention.q_proj, block.attn.attention.k_proj,
                          block.attn.attention.v_proj):
                layer.bias = torch.nn.Parameter(torch.randn(16)/10)
            for name, param in block.named_parameters():
                if name.endswith('bias'):
                    param.copy_(torch.randn_like(param)/10)
            for tokens in (1, 7):
                x = torch.randn(tokens, 16)
                expected = block(x.unsqueeze(0), use_cache=False)[0][0]
                for backend, node in devices:
                    actual = infer_block(block, config, x, backend=backend, node=node)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                    print('decoder activation={} type={} tokens={} backend={} node={} '
                          'max_abs_error={:.3g} PASS'.format(
                              activation, block.attn.attention_type, tokens,
                              backend, node, (actual-expected).abs().max().item()), flush=True)


def check_trained(devices, count):
    model, tokenizer = load_model()
    print('trained_model=TinyStories-1M unique_parameters={}'.format(
        sum(p.numel() for p in model.parameters())), flush=True)
    prompts = ['Once upon a time there was a little girl', 'One day, a small dog']
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        layer_outputs = []
        handles = [block.register_forward_hook(
            lambda module, inputs, output: layer_outputs.append(output[0][0].clone()))
            for block in model.transformer.h]
        try:
            expected = model(ids.unsqueeze(0), use_cache=False).logits[0]
        finally:
            for handle in handles:
                handle.remove()
        for backend, node in devices:
            native = GPTNeoNative(model, node, backend)
            actual = native.logits(ids, check_layers=layer_outputs)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
            print('trained_logits prompt_tokens={} backend={} node={} '
                  'max_abs_error={:.3g} argmax_match=True PASS'.format(
                      len(ids), backend, node, (actual-expected).abs().max().item()), flush=True)
    # A real checkpoint sequence crosses the configured 256-token local window.
    ids = torch.tensor([int(i*37 % model.config.vocab_size) for i in range(259)])
    expected = model(ids.unsqueeze(0), use_cache=False).logits[0, -1:]
    for backend, node in devices:
        actual = GPTNeoNative(model, node, backend).logits(ids, last_only=True)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        print('trained_local_boundary tokens=259 backend={} node={} '
              'max_abs_error={:.3g} PASS'.format(
                  backend, node, (actual-expected).abs().max().item()), flush=True)
    ids = tokenizer.encode(prompts[0], return_tensors='pt')[0]
    def cpu_last(tokens):
        return model(tokens.unsqueeze(0), use_cache=False).logits[0, -1]
    cpu_last(ids)
    start = time.perf_counter()
    reference, ref_ids = greedy_generate(cpu_last, ids, count, model.config.eos_token_id)
    cpu_seconds = time.perf_counter()-start
    print(json.dumps(dict(event='generation', backend='torch', threads=torch.get_num_threads(),
                          generated_ids=ref_ids, seconds=cpu_seconds,
                          tokens_per_second=len(ref_ids)/cpu_seconds,
                          text=tokenizer.decode(reference, skip_special_tokens=True))), flush=True)
    for backend, node in devices:
        native = GPTNeoNative(model, node, backend)
        native.logits(ids, last_only=True)
        start = time.perf_counter()
        output, generated = greedy_generate(
            lambda tokens: native.logits(tokens, last_only=True)[0], ids, count,
            model.config.eos_token_id)
        seconds = time.perf_counter()-start
        assert generated == ref_ids and torch.equal(output, reference)
        print(json.dumps(dict(event='generation', backend=backend, node=node,
                              generated_ids=generated, seconds=seconds,
                              tokens_per_second=len(generated)/seconds,
                              ids_match=True, result='PASS')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, nargs='*', default=[1, 2, 3])
    parser.add_argument('--new-tokens', type=int, default=16)
    parser.add_argument('--random-only', action='store_true')
    args = parser.parse_args()
    if any(n < 0 for n in args.nodes) or not 1 <= args.new_tokens <= 64:
        parser.error('invalid node or generation count')
    torch.set_num_threads(1)
    devices = [('cpu', 0)] + [('fast', n) for n in args.nodes]
    with torch.inference_mode():
        check_random(devices)
        if not args.random_only:
            check_trained(devices, args.new_tokens)
