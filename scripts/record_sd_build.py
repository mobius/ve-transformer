"""Capture the isolated image executable and exact adaptation inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('cpu','ve'),required=True);args=p.parse_args()
    source=ROOT/'build/vendor/stable-diffusion.cpp'
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    ggml_revision=subprocess.check_output(['git','-C',str(source/'ggml'),'rev-parse','HEAD'],text=True).strip()
    if revision!='a1ded76da5818803fca97a3b433669ef727d32cf' or ggml_revision!='89c4413f5da6fb20cc796f16033d37f129be81fd':raise RuntimeError('unexpected framework revision')
    files=['scripts/build_sd_turbo.sh','scripts/prepare_sd_overlay.py','cmake/nec-sd-overrides.cmake','src/ve_sd_image.cpp',
           'build/sd-overlay/sd-ggml.c','build/sd-overlay/sd-gguf.cpp','build/sd-overlay/sd-image.cpp',
           'build/sd-overlay/sd-diffusion-engine.cpp','build/sd-overlay/sd-ggml-extend.cpp','build/sd-overlay/sd_validation.h',
           'build/sd-overlay/sd-convert-disabled.cpp',
           'build/sd-overlay/sd-detailer-disabled.cpp',
           'build/sd-overlay/sd-model-builders.cpp',
           'build/sd-overlay/sd-video-disabled.cpp',
           'build/sd-overlay/include/model/common/ggml_block.hpp','build/sd-overlay/include/model/common/block.hpp',
           'build/sd-overlay/include/model/diffusion/unet.hpp',
           'build/sd-'+args.backend+'/bin/sd-cli']
    report={'framework_revision':revision,'ggml_revision':ggml_revision,'backend':args.backend,
            'model_conversion_enabled':args.backend!='ve','activation':'erf GELU',
            'automatic_detailing_enabled':args.backend!='ve',
            'runner_scope':'SD2 and standard VAE' if args.backend=='ve' else 'upstream factories',
            'video_generation_enabled':args.backend!='ve',
            'orchestration_optimization':'O0 and no inlining' if args.backend=='ve' else 'O1',
            'numerical_kernel_optimization':'O1; NLC library unchanged',
            'image_scope':'single 512x512 SD2 text-to-image, Euler, no guidance mixing or optional image features' if args.backend=='ve' else 'upstream image pipeline',
            'precision':'FP32 weights and FP32 convolution input packing','compiler_duty_percent':int(os.environ.get('SD_BUILD_DUTY_PERCENT','75')),
            'sha256':{name:sha(ROOT/name) for name in files}}
    out=ROOT/('build/sd-'+args.backend+'/manifest.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print('Image build manifest captured:',out.relative_to(ROOT))

if __name__=='__main__':main()
