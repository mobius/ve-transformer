"""Independent guarded CPU FP32 SD-Turbo reference and fixed initial noise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
PROMPTS=['A red cube on a blue table, studio lighting.',
         'A yellow ceramic cup beside a green apple, product photography.',
         'A wooden cabin beside a mountain lake at sunrise.',
         'A close-up photograph of woven blue fabric.',
         'A white sailboat on calm water, clear sky.',
         'A small robot holding a red balloon, illustration.']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':
        raise RuntimeError('temperature supervision required')
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=int,choices=range(1,7),default=1)
    parser.add_argument('--steps',type=int,choices=(1,4),default=1)
    parser.add_argument('--size',type=int,choices=(256,512),default=512)
    args=parser.parse_args()
    # Local-only loading: no implicit downloads or global framework caches.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      HF_HOME=str(ROOT/'build/cache/huggingface'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    from diffusers import StableDiffusionPipeline
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    model=ROOT/'build/models/sd-turbo'
    provenance=json.loads((model/'provenance.json').read_text())
    if not provenance['completed'] or provenance['revision']!='b261bac6fd2cf515557d5d0707481eafa0485ec2':
        raise RuntimeError('completed pinned official weights required')
    for name,entry in provenance['files'].items():
        if sha(model/name)!=entry['sha256']:raise RuntimeError('reference source checksum mismatch')
    folder=ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-sd-reference',time.gmtime())
    folder.mkdir()
    report=dict(completed=False,model_revision=provenance['revision'],steps=args.steps,size=args.size,
                precision='CPU FP32',threads=1,test_sha256=sha(Path(__file__)),cases=[],
                cpu_duty_percent=int(os.environ.get('SD_REFERENCE_DUTY_PERCENT','100')),
                timing_scope='reference execution including any explicit CPU duty throttling; not an optimized CPU benchmark')
    def save():(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save();(folder/'export_sd_reference.py').write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    pipe=StableDiffusionPipeline.from_pretrained(model,torch_dtype=torch.float32,
                                                 local_files_only=True,safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    report['load_seconds']=time.perf_counter()-start
    for index,prompt in enumerate(PROMPTS[:args.cases]):
        print('CPU reference case',index,'steps',args.steps,'size',args.size,flush=True)
        case=folder/('case%d'%index);case.mkdir();files={}
        def array(name,value):
            if isinstance(value,torch.Tensor):value=value.detach().cpu().float().numpy()
            value=np.ascontiguousarray(value,dtype=np.float32)
            if not np.isfinite(value).all():raise RuntimeError('nonfinite reference array')
            path=case/(name+'.npy');np.save(path,value)
            raw=case/(name+'.f32');value.tofile(raw)
            files[name]={'shape':list(value.shape),'sha256':sha(raw),'bytes':raw.stat().st_size}
        with torch.inference_mode():
            started=time.perf_counter()
            ids=pipe.tokenizer(prompt,padding='max_length',max_length=pipe.tokenizer.model_max_length,
                               truncation=True,return_tensors='pt').input_ids
            embeddings=pipe.text_encoder(ids,return_dict=False)[0]
            text_seconds=time.perf_counter()-started;array('embeddings',embeddings)
            generator=torch.Generator(device='cpu').manual_seed(2026+index)
            noise=torch.randn((1,4,args.size//8,args.size//8),generator=generator,dtype=torch.float32)
            array('noise',noise)
            pipe.scheduler.set_timesteps(args.steps,device='cpu')
            array('timesteps',pipe.scheduler.timesteps);array('sigmas',pipe.scheduler.sigmas)
            latent=noise*pipe.scheduler.init_noise_sigma
            stage=[]
            for step,timestep in enumerate(pipe.scheduler.timesteps):
                model_input=pipe.scheduler.scale_model_input(latent,timestep)
                array('step%d-input'%step,model_input)
                started=time.perf_counter()
                epsilon=pipe.unet(model_input,timestep,encoder_hidden_states=embeddings,return_dict=False)[0]
                elapsed=time.perf_counter()-started;array('step%d-epsilon'%step,epsilon)
                latent=pipe.scheduler.step(epsilon,timestep,latent,return_dict=False)[0]
                array('step%d-latent'%step,latent);stage.append(elapsed)
            started=time.perf_counter()
            image=pipe.vae.decode(latent/pipe.vae.config.scaling_factor,return_dict=False)[0]
            vae_seconds=time.perf_counter()-started;array('decoded',image)
            pipe.image_processor.postprocess(image,output_type='pil')[0].save(case/'image.png')
        report['cases'].append(dict(case=index,seed=2026+index,text_seconds=text_seconds,
                                    unet_seconds=stage,vae_seconds=vae_seconds,arrays=files))
        save()
    report.update(completed=True,torch_version=torch.__version__)
    save();print('CPU reference complete:',folder.relative_to(ROOT),flush=True)

if __name__=='__main__':main()
