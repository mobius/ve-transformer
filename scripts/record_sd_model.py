"""Summarize pinned image tensors without publishing raw upstream configs."""
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'build/models/sd-turbo'
    source=json.loads((folder/'provenance.json').read_text())
    if not source['completed']:raise RuntimeError('incomplete model preparation')
    components={}
    for name,entry in source['files'].items():
        if not name.endswith('.safetensors'):continue
        with (folder/name).open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if length>16*1024**2:raise RuntimeError('oversized tensor header')
            header=json.loads(f.read(length))
        count=0;parameters=0;payload=0;dtypes={}
        for key,value in header.items():
            if key=='__metadata__':continue
            count+=1
            n=1
            for dim in value['shape']:n*=dim
            parameters+=n
            begin,end=value['data_offsets'];payload+=end-begin
            dtype=value['dtype'];dtypes[dtype]=dtypes.get(dtype,0)+n
        if set(dtypes)!={'F32'}:raise RuntimeError('official FP32 reference contains another dtype')
        components[name]={'tensors':count,'parameters':parameters,'payload_bytes':payload,'dtypes_by_parameter_count':dtypes,
                          'file_sha256':entry['sha256']}
    report={'model_repo':source['repo'],'model_revision':source['revision'],'components':components,
            'total_parameters':sum(c['parameters'] for c in components.values()),
            'weight_payload_bytes':sum(c['payload_bytes'] for c in components.values()),
            'memory_scope':'weights only; graph intermediates, working buffers, allocators and runtime excluded',
            'configuration':{}}
    for component,keys in [('unet',('sample_size','in_channels','out_channels','cross_attention_dim','block_out_channels')),
                           ('text_encoder',('hidden_size','num_hidden_layers','max_position_embeddings')),
                           ('vae',('scaling_factor','in_channels','out_channels','latent_channels')),
                           ('scheduler',('prediction_type','timestep_spacing','num_train_timesteps','beta_start','beta_end','beta_schedule'))]:
        path=folder/component/('scheduler_config.json' if component=='scheduler' else 'config.json')
        config=json.loads(path.read_text());report['configuration'][component]={k:config[k] for k in keys}
    from record_qwen36_mtp import safe
    safe(report)
    stamp=(ROOT/'build/sd-stage-stamp').read_text()
    out=ROOT/'docs/results'/(stamp+'-sd-turbo-model.json')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('Pinned FP32 model summary:',out.relative_to(ROOT))
    print('Parameters:',report['total_parameters'],'weight payload GiB:',report['weight_payload_bytes']/1024**3)

if __name__=='__main__':main()
