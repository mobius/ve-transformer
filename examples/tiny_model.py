"""A tiny, untrained two-layer model: integer tokens -> output logits.

Embedding and output head execute on host; Transformer layers execute on VE.
This validates architecture and exported weights, not language-model quality.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from torch import nn
from ve_transformer import infer_layer


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 32)
        self.position = nn.Embedding(64, 32)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(32, 4, 64, dropout=0, activation='relu',
                                       norm_first=True, bias=False)
            for _ in range(2)])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 32, bias=False)

    def features(self, tokens):
        return self.embedding(tokens)+self.position(torch.arange(tokens.numel()))

    def forward(self, tokens):
        x = self.features(tokens)
        mask = torch.triu(torch.full((len(tokens), len(tokens)), float('-inf'),
                                    dtype=x.dtype), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.head(self.norm(x))

    @torch.inference_mode()
    def forward_ve(self, tokens, node, backend='nlc'):
        x = self.features(tokens)
        for layer in self.layers:
            x = infer_layer(layer, x, node=node, backend=backend, causal=True)
        return self.head(self.norm(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['nlc', 'fast'], default='nlc')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = TinyTransformer().eval()
    tokens = torch.tensor([1, 4, 2, 8, 3, 5, 13, 6], dtype=torch.long)
    with torch.inference_mode():
        expected = model.double()(tokens)
        model.float()
        for node in (1, 2, 3):
            actual = model.forward_ve(tokens, node, args.backend)
            torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=1e-4)
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
            print('two-layer model node={} max_logits_error={:.3g} argmax_match=True PASS'
                  .format(node, (actual.double()-expected).abs().max().item()))
    print('Untrained model: output IDs do not represent useful generated text.')
