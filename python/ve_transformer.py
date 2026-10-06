"""Inference-only bridge for a restricted PyTorch TransformerEncoderLayer.

This exports weights and runs a native process; it is not a torch device backend.
Only pre-norm, ReLU, bias-free linear layers, epsilon 1e-5, single CPU float32
sequence are supported. Input/output and parameters are transferred per call.
"""
from pathlib import Path
import os
import struct
import subprocess
import tempfile
import time
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def export_layer(layer, x, path, *, causal=False):
    if not isinstance(layer, nn.TransformerEncoderLayer):
        raise TypeError('expected TransformerEncoderLayer')
    if layer.training or not layer.norm_first:
        raise ValueError('eval mode and norm_first=True are required')
    if layer.activation is not F.relu and not isinstance(layer.activation, nn.ReLU):
        raise ValueError('only ReLU is supported')
    if layer.norm1.eps != 1e-5 or layer.norm2.eps != 1e-5:
        raise ValueError('LayerNorm epsilon must be 1e-5')
    if not layer.norm1.elementwise_affine or not layer.norm2.elementwise_affine:
        raise ValueError('affine LayerNorm is required')
    attn = layer.self_attn
    if (attn.in_proj_bias is not None or attn.out_proj.bias is not None
            or layer.linear1.bias is not None or layer.linear2.bias is not None
            or attn.bias_k is not None or attn.bias_v is not None
            or attn.add_zero_attn):
        raise ValueError('linear biases and extra attention positions are unsupported')
    if (x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 2
            or x.requires_grad):
        raise ValueError('input must be a CPU float32 [tokens, width] inference tensor')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in layer.parameters()):
        raise ValueError('parameters must be CPU float32')
    t, d = x.shape
    h, f = attn.num_heads, layer.linear1.out_features
    if not (1 <= t <= 1024 and 1 <= d <= 2048 and 1 <= f <= 8192
            and d == attn.embed_dim and d % h == 0):
        raise ValueError('shape outside native format bounds')
    if attn.in_proj_weight is None or tuple(attn.in_proj_weight.shape) != (3*d, d):
        raise ValueError('packed equal-width Q/K/V weights required')
    q, k, v = attn.in_proj_weight.detach().chunk(3, dim=0)
    tensors = [q.T, k.T, v.T, attn.out_proj.weight.detach().T,
               layer.linear1.weight.detach().T, layer.linear2.weight.detach().T]
    for norm in (layer.norm1, layer.norm2):
        tensors += [norm.weight.detach(),
                    norm.bias.detach() if norm.bias is not None else torch.zeros(d)]
    tensors.append(x.detach())
    arrays = [np.asarray(a.contiguous().numpy(), dtype='<f4') for a in tensors]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite tensors are unsupported')
    with open(path, 'wb') as handle:
        handle.write(b'VTF32V1\0')
        handle.write(struct.pack('<5I', t, d, h, f, int(bool(causal))))
        for a in arrays:
            handle.write(a.tobytes(order='C'))


@torch.inference_mode()
def infer_layer(layer, x, *, node=1, backend='nlc', causal=False):
    return run_native_export(lambda source: export_layer(layer, x, source, causal=causal),
                             x, node=node, backend=backend)


def run_native_export(exporter, x, *, node=1, backend='fast', timings=None):
    """Run a validated exporter in a bounded, per-call native process."""
    if backend not in ('cpu', 'base', 'nlc', 'fast'):
        raise ValueError('backend must be cpu, base, nlc or fast')
    if type(node) is not int or node < 0:
        raise ValueError('node must be a nonnegative integer')
    binary = {'cpu': 'infer-cpu', 'base': 'infer-ve', 'nlc': 'infer-ve-nlc',
              'fast': 'infer-ve-fast'}[backend]
    binary = ROOT/'build'/binary
    if not binary.is_file():
        raise FileNotFoundError('build native inference binaries with make first')
    env = os.environ.copy()
    libraries = ['/opt/nec/ve/ncc/5.4.1/lib', '/opt/nec/ve/nfort/5.4.1/lib',
                 '/opt/nec/ve/nlc/3.1.0/lib']
    if env.get('VE_LD_LIBRARY_PATH'):
        libraries.append(env['VE_LD_LIBRARY_PATH'])
    env['VE_LD_LIBRARY_PATH'] = ':'.join(libraries)
    env.setdefault('OMP_NUM_THREADS', '8')
    env['OMP_DYNAMIC'] = 'FALSE'
    with tempfile.TemporaryDirectory(prefix='inference-', dir=ROOT/'build') as tmp:
        source, result = Path(tmp)/'request.vtf', Path(tmp)/'output.f32'
        begin = time.perf_counter() if timings is not None else 0
        exporter(source)
        exported = time.perf_counter() if timings is not None else 0
        command = [str(binary), str(source), str(result)]
        if backend != 'cpu':
            command = ['/opt/nec/ve/bin/ve_exec', '-N', str(node)] + command
        if timings is not None:
            command.append('--profile')
        process = subprocess.run(command, env=env, check=True, timeout=60, capture_output=True)
        finished = time.perf_counter() if timings is not None else 0
        data = result.read_bytes()
        if len(data) != x.numel()*4:
            raise RuntimeError('native output length mismatch')
        output = np.frombuffer(data, dtype='<f4').reshape(tuple(x.shape)).copy()
        if not np.isfinite(output).all():
            raise RuntimeError('nonfinite native output')
        if timings is not None:
            match = re.fullmatch(rb'native_forward_ms=([0-9.]+)\n', process.stdout)
            if not match:
                raise RuntimeError('unexpected native profile output')
            timings.append(dict(export_ms=(exported-begin)*1000,
                                process_ms=(finished-exported)*1000,
                                read_ms=(time.perf_counter()-finished)*1000,
                                compute_ms=float(match.group(1))))
        return torch.from_numpy(output)
