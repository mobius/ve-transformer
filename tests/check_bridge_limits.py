"""Reject framework semantics the native layer cannot implement."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from torch import nn
from ve_transformer import export_layer


def layer(**changes):
    options = dict(d_model=8, nhead=2, dim_feedforward=16, dropout=0,
                   norm_first=True, bias=False)
    options.update(changes)
    return nn.TransformerEncoderLayer(**options).eval()


root = Path(__file__).resolve().parents[1]
x = torch.zeros(4, 8)
cases = [(layer().train(), x), (layer(norm_first=False), x),
         (layer(activation='gelu'), x), (layer(bias=True), x),
         (layer(layer_norm_eps=1e-4), x), (layer(), x.double()),
         (layer(), x.unsqueeze(0)), (layer(), x.clone().requires_grad_()),
         (layer(), torch.full((4, 8), float('nan')))]
without_affine = layer()
without_affine.norm1 = nn.LayerNorm(8, elementwise_affine=False)
cases.append((without_affine, x))
with tempfile.TemporaryDirectory(dir=root/'build') as tmp:
    for model, value in cases:
        try:
            export_layer(model, value, Path(tmp)/'request.vtf')
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported semantics silently accepted')
print('PyTorch bridge semantic restrictions: PASS')
