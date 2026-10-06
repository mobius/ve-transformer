"""Exercise ping-pong output parity using an odd block count and varying lengths."""
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from resident_ve import GPTNeoResident

parser=argparse.ArgumentParser()
parser.add_argument('--node',type=int,default=1)
args=parser.parse_args()
if args.node<0: parser.error('node must be nonnegative')
torch.set_num_threads(1)
torch.manual_seed(1203)
config=GPTNeoConfig(vocab_size=32,hidden_size=16,num_layers=3,num_heads=4,
                   intermediate_size=31,max_position_embeddings=16,
                   attention_types=[[['global','local','global'],1]],window_size=3,
                   activation_function='gelu_new',resid_dropout=0,embed_dropout=0,
                   attention_dropout=0)
config._attn_implementation='eager'
model=GPTNeoForCausalLM(config).eval()
with torch.inference_mode():
    for backend in ('cpu','fast'):
        with GPTNeoResident(model,args.node,backend,capacity=16,threads=1) as resident:
            for tokens in (1,7,2,9,7):
                ids=torch.tensor([(i*3+tokens)%32 for i in range(tokens)])
                expected=model(ids.unsqueeze(0),use_cache=False).logits[0]
                actual=resident.logits(ids)
                torch.testing.assert_close(actual,expected,atol=2e-5,rtol=1e-4)
                print('resident odd_layers=3 backend={} tokens={} PASS'.format(backend,tokens))
