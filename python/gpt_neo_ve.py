"""Restricted pretrained GPT-Neo inference: blocks on native CPU/VE, head on host.

No external masks, batching, training or generation cache. GELU tanh and both
causal attention types preserve GPT-Neo's unscaled attention semantics.
"""
from pathlib import Path
import hashlib
import json
import struct

import numpy as np
import torch
from ve_transformer import ROOT, run_native_export

MODEL_REVISION = '77f1b168e219585646439073245fe87e56b3023e'
MODEL_DIR = ROOT/'build/models/tinystories-1m'


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    manifest = json.loads((MODEL_DIR/'provenance.json').read_text())
    if (manifest['model'] != 'roneneldan/TinyStories-1M'
            or manifest['revision'] != MODEL_REVISION):
        raise ValueError('unexpected checkpoint provenance')
    for name, expected in manifest['sha256'].items():
        if Path(name).name != name:
            raise ValueError('invalid manifest file name')
        if hashlib.sha256((MODEL_DIR/name).read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint hash mismatch: '+name)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False,
        use_safetensors=False, weights_only=True, attn_implementation='eager',
        dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False)
    return model, tokenizer


def export_block(block, config, x, path):
    from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoSelfAttention
    if not isinstance(block, GPTNeoBlock) or config.model_type != 'gpt_neo':
        raise TypeError('expected a GPTNeoBlock')
    if block.training or config.activation_function not in ('gelu_new', 'relu'):
        raise ValueError('eval and gelu_new/relu required')
    if (x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 2
            or x.requires_grad):
        raise ValueError('CPU float32 [tokens,width] inference tensor required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in block.parameters()):
        raise ValueError('CPU float32 parameters required')
    t, d = x.shape
    attention = block.attn.attention
    if type(attention) is not GPTNeoSelfAttention or config._attn_implementation != 'eager':
        raise ValueError('only the verified eager GPT-Neo attention implementation is supported')
    h, f = attention.num_heads, block.mlp.c_fc.out_features
    if not (1 <= t <= 1024 and 1 <= d <= 2048 and d == config.hidden_size
            and d == attention.embed_dim and h > 0 and d % h == 0 and 1 <= f <= 8192):
        raise ValueError('shape outside native bounds')
    if block.attn.attention_type not in ('global', 'local'):
        raise ValueError('unsupported attention type')
    window = config.window_size if block.attn.attention_type == 'local' else 0
    if type(window) is not int or not 0 <= window <= 1024:
        raise ValueError('unsupported local window')
    if block.attn.attention_type == 'local' and not window:
        raise ValueError('local attention requires a positive window')
    epsilon = block.ln_1.eps
    if not (0 < epsilon < 1 and epsilon == block.ln_2.eps
            and epsilon == config.layer_norm_epsilon):
        raise ValueError('inconsistent normalization epsilon')
    if (not block.ln_1.elementwise_affine or not block.ln_2.elementwise_affine
            or block.ln_1.bias is None or block.ln_2.bias is None):
        raise ValueError('affine LayerNorm required')
    linears = [attention.q_proj, attention.k_proj, attention.v_proj,
               attention.out_proj, block.mlp.c_fc, block.mlp.c_proj]
    if any(tuple(layer.weight.shape) != shape for layer, shape in zip(
            linears, [(d,d)]*4+[(f,d),(d,f)])):
        raise ValueError('unexpected projection shape')
    tensors = [layer.weight.detach().T for layer in linears]
    for norm in (block.ln_1, block.ln_2):
        tensors.extend([norm.weight.detach(), norm.bias.detach()])
    tensors.extend([layer.bias.detach() if layer.bias is not None
                    else torch.zeros(layer.out_features) for layer in linears])
    tensors.append(x.detach())
    arrays = [np.asarray(a.contiguous().numpy(), dtype='<f4') for a in tensors]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite tensors')
    with open(path, 'wb') as handle:
        handle.write(b'VTF32V2\0')
        handle.write(struct.pack('<8If', t, d, h, f, 1,
                                 int(config.activation_function == 'gelu_new'), 1,
                                 window, epsilon))
        for a in arrays:
            handle.write(a.tobytes(order='C'))


@torch.inference_mode()
def infer_block(block, config, x, *, node=1, backend='fast', timings=None):
    return run_native_export(lambda path: export_block(block, config, x, path),
                             x, node=node, backend=backend, timings=timings)


class GPTNeoNative:
    def __init__(self, model, node=1, backend='fast'):
        if model.config.model_type != 'gpt_neo' or model.training:
            raise ValueError('eval GPT-Neo model required')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('CPU float32 model required')
        self.model, self.node, self.backend = model, node, backend
        self.trace = None

    @torch.inference_mode()
    def features(self, ids):
        if (ids.ndim != 1 or ids.device.type != 'cpu' or ids.dtype != torch.long
                or not 1 <= ids.numel() <= min(1024, self.model.config.max_position_embeddings)
                or ids.min().item() < 0 or ids.max().item() >= self.model.config.vocab_size):
            raise ValueError('bounded one-dimensional CPU token IDs required')
        position = torch.arange(ids.numel())
        return self.model.transformer.wte(ids)+self.model.transformer.wpe(position)

    @torch.inference_mode()
    def logits(self, ids, *, last_only=False, check_layers=None):
        x = self.features(ids)
        for index, block in enumerate(self.model.transformer.h):
            x = infer_block(block, self.model.config, x, node=self.node, backend=self.backend,
                            timings=self.trace)
            if check_layers is not None:
                torch.testing.assert_close(x, check_layers[index], atol=2e-4, rtol=2e-4)
        if last_only:
            x = x[-1:]
        return self.model.lm_head(self.model.transformer.ln_f(x))


@torch.inference_mode()
def greedy_generate(forward_last, ids, count, eos_token_id):
    if not 1 <= count <= 64 or ids.numel()+count > 1024:
        raise ValueError('generation exceeds bounded context or step count')
    result = ids.clone()
    generated = []
    for _ in range(count):
        scores = forward_last(result)
        if not torch.isfinite(scores).all():
            raise RuntimeError('nonfinite generation scores')
        token = int(scores.reshape(-1).argmax())
        generated.append(token)
        result = torch.cat([result, torch.tensor([token], dtype=torch.long)])
        if token == eos_token_id:
            break
    return result, generated
