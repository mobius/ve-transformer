"""Publish bounded stage-validation evidence, excluding images and raw logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from record_qwen36_mtp import safe, thermal

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-log',type=Path,required=True)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--validation-log',type=Path)
    parser.add_argument('--operators-log',type=Path)
    parser.add_argument('--operators-source',type=Path,default=ROOT/'tests/check_sd_ops.cpp')
    parser.add_argument('--operators-framework',choices=('compact','modern'),default='modern')
    args=parser.parse_args()
    reference=args.reference.resolve();reference.relative_to(ROOT/'build')
    cpu=json.loads((reference/'summary.json').read_text())
    if not cpu['completed'] or cpu['size']!=512:raise RuntimeError('complete 512 reference required')
    report={'status':'cpu_reference_complete_ve_pending','default_adopted':False,
            'model_repo':'stabilityai/sd-turbo','model_revision':cpu['model_revision'],
            'architecture':'dense distilled SD2.1: CLIP, U-Net, VAE',
            'reference_artifacts':str(reference.relative_to(ROOT)),
            'reference':cpu,'reference_temperature':thermal(args.reference_log),
            'timing_scope':'correctness validation including model loading and trace export; no steady-state performance claim',
            'physical_hbm_peak_measured':False,
            'quality_scope':'numerical agreement and generated-image inspection; no quality benchmark'}
    if args.operators_log:
        log=args.operators_log.read_text()
        match=re.search(r'IMAGE_OPERATOR_PASS cases=(\d+);',log)
        checks=re.findall(r'^(.+) relative_l2=([0-9.e+-]+) max_abs=([0-9.e+-]+) PASS$',log,re.M)
        if not match or len(checks)!=int(match[1]):raise RuntimeError('completed independent operator checks required')
        report.update(status='cpu_reference_and_ve_operators_complete_full_image_pending',
                      operators={'cases':int(match[1]),'relative_l2_limit':1e-4,
                                 'framework_variant':args.operators_framework,
                                 'checks':[{'name':name,'relative_l2':float(relative),'max_abs_error':float(maximum)}
                                           for name,relative,maximum in checks],
                                 'source_sha256':hashlib.sha256(args.operators_source.read_bytes()).hexdigest(),
                                 'temperature':thermal(args.operators_log)})
    if args.validation:
        if not args.validation_log or not args.operators_log:raise RuntimeError('hardware evidence logs required')
        native=json.loads((args.validation/'summary.json').read_text())
        if not native['completed'] or native['reference_artifacts']!=str(reference.relative_to(ROOT)):
            raise RuntimeError('matching completed VE validation required')
        base=ROOT/('build/sd-baseline-ve' if native.get('framework_variant')=='compact' else 'build/sd-ve')
        manifest=json.loads((base/'manifest.json').read_text())
        digest=hashlib.sha256((base/'bin'/('sd' if native.get('framework_variant')=='compact' else 'sd-cli')).read_bytes()).hexdigest()
        if native['binary_sha256']!=digest or manifest['backend']!='ve':raise RuntimeError('VE executable differs')
        report.update(status='single_ve_full_image_stages_verified',validation=native,
                      validation_temperature=thermal(args.validation_log),build=manifest,
                      execution_scope='CLIP, U-Net, VAE and scheduler in one native VE process; host launch and file validation',
                      validation_artifacts=str(args.validation.resolve().relative_to(ROOT)))
    safe(report)
    path=ROOT/'docs/results'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-sd-turbo-validation.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Public image validation evidence:',path.relative_to(ROOT))

if __name__=='__main__':main()
