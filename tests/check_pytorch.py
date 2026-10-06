"""Exercise exported standard PyTorch weights on CPU and all three VE slots."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from torch import nn
from ve_transformer import infer_layer

parser = argparse.ArgumentParser()
parser.add_argument('--fast-only', action='store_true')
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(20261002)
for t, d, heads, hidden in [(7, 15, 3, 31), (32, 64, 4, 128), (128, 256, 8, 1024)]:
    layer = nn.TransformerEncoderLayer(d, heads, hidden, dropout=0,
                                      activation='relu', norm_first=True, bias=False).eval()
    x = torch.randn(t, d)
    for causal in (False, True):
        mask = torch.triu(torch.full((t, t), float('-inf')), diagonal=1) if causal else None
        with torch.inference_mode():
            # Float64 version of the same float32 parameters/input is the oracle.
            layer.double()
            expected = layer(x.double(), src_mask=mask.double() if mask is not None else None)
            layer.float()
        devices = [('cpu', 0)] + [(b, n) for b in ('base', 'nlc') for n in (1, 2, 3)]
        if args.fast_only:
            devices = [('fast', n) for n in (1, 2, 3)]
        for backend, node in devices:
            actual = infer_layer(layer, x, node=node, backend=backend, causal=causal)
            torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=1e-4)
            error = (actual.double()-expected).abs().max().item()
            print('pytorch shape={}x{} causal={} backend={} node={} max_abs_error={:.3g} PASS'
                  .format(t, d, causal, backend, node, error), flush=True)
