"""Reuse fixed CPU traces to test isolated precision builds on the real model."""
import argparse
import json
import shutil
import time
from check_qwen_model import ROOT,PROMPTS,chat,run,compare,verify_model

parser=argparse.ArgumentParser()
parser.add_argument('--backend',choices=['quant','nlc'],required=True)
args=parser.parse_args()
reference_folder=ROOT/'build/results'/('20261002T133204Z-qwen36' if args.backend=='quant'
                                      else '20261002T134602Z-qwen36')
reference=json.loads((reference_folder/'case0-cpu.jsonl').read_text())
verify_model()
folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-qwen-precision-'+args.backend,time.gmtime())
folder.mkdir(parents=True)
shutil.copyfile(reference_folder/'case0-cpu-0.f32',folder/'reference-0.f32')
backend='ve' if args.backend=='quant' else 'nlc'
binary=ROOT/'build/qwen-ve-precision'/('qwen-infer-ve' if args.backend=='quant' else 'qwen-infer-ve-nlc')
actual=run(folder,'precision',backend,chat(PROMPTS[0]),count=12,
           forced=reference_folder/'case0-cpu.tokens',trace=True,binary_override=binary)[0]
report=compare(folder,'reference','precision',reference,actual,12)
(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('artifacts='+str(folder.relative_to(ROOT)),flush=True)
