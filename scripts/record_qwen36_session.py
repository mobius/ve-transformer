"""Publish sanitized resident-session acceptance, preserving private traces."""
import argparse
import json
from pathlib import Path
from record_qwen36_mtp import ROOT,safe,thermal

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--build-log',type=Path,required=True)
    parser.add_argument('--stamp',required=True)
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z',args.stamp):raise RuntimeError('invalid UTC stamp')
    report=json.loads((args.validation/'summary.json').read_text())
    if not report['completed'] or [m['mode'] for m in report['modes']]!=['ordinary','fixed','auto']:
        raise RuntimeError('completed three-mode acceptance required')
    for mode in report['modes']:
        if not mode['protocol_rejection'] or [r['case'] for r in mode['requests']]!=[1,2,0,1]:
            raise RuntimeError('cross-request and protocol rejection acceptance required')
    report.update(status='resident_session_ve_verified',runtime_slot=1,
                  validation_artifacts=str(args.validation.resolve().relative_to(ROOT)),
                  temperature=thermal(args.log),build_temperature=thermal(args.build_log),
                  performance_scope='correctness screening with one warm repeated code request per mode; not ABBA; no performance adoption',
                  state_scope='per-request target and draft state cleared; immutable weights retained',
                  automatic_policy='one scalar calibration; at least three speculative rounds; disable for remaining request when unit cost exceeds scalar reference',
                  limits='greedy text only; context 512; no vision, other slots, maximum trained context or new full-logit oracle')
    safe(report)
    out=ROOT/'docs/results'/(args.stamp+'-qwen36-mtp-session.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Resident acceptance published:',out.relative_to(ROOT))

if __name__=='__main__':main()
