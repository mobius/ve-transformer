"""Download only the pinned official FP32 Diffusers SD-Turbo layout locally."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='stabilityai/sd-turbo'
REVISION='b261bac6fd2cf515557d5d0707481eafa0485ec2'
FILES=['model_index.json','scheduler/scheduler_config.json','text_encoder/config.json',
       'tokenizer/special_tokens_map.json','tokenizer/tokenizer_config.json',
       'tokenizer/vocab.json','tokenizer/merges.txt','unet/config.json','vae/config.json',
       'text_encoder/model.safetensors','unet/diffusion_pytorch_model.safetensors',
       'vae/diffusion_pytorch_model.safetensors']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    dest=ROOT/'build/models/sd-turbo'
    dest.mkdir(parents=True,exist_ok=True)
    api='https://huggingface.co/api/models/'+REPO+'/revision/'+REVISION+'?blobs=true'
    with urllib.request.urlopen(api,timeout=30) as response:info=json.load(response)
    if info['sha']!=REVISION:raise RuntimeError('official revision mismatch')
    siblings={entry['rfilename']:entry for entry in info['siblings']}
    manifest={'repo':REPO,'revision':REVISION,'format':'official FP32 Diffusers layout','files':{},'completed':False}
    for name in FILES:
        entry=siblings[name];path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
        size=entry['size'];expected=entry.get('lfs',{}).get('sha256')
        if path.exists() and path.stat().st_size!=size:raise RuntimeError('existing file size mismatch')
        if path.exists() and path.stat().st_size==size and (expected is None or digest(path)==expected):
            if expected is None:
                blob=path.read_bytes()
                if hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()!=entry['blobId']:
                    path.unlink()
        if not path.exists():
            part=path.with_name(path.name+'.part')
            offset=part.stat().st_size if part.exists() else 0
            if offset>size:raise RuntimeError('partial file exceeds source size')
            print('Downloading official file:',name,'bytes',size,'resume',offset,flush=True)
            with part.open('ab') as output:
                while offset<size:
                    end=min(size-1,offset+32*1024**2-1)
                    url='https://huggingface.co/'+REPO+'/resolve/'+REVISION+'/'+name+'?download=true&offset='+str(offset)
                    for attempt in range(3):
                        try:
                            req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(offset,end)})
                            with urllib.request.urlopen(req,timeout=30) as response:
                                if size>32*1024**2:
                                    wanted='bytes %d-%d/%d'%(offset,end,size)
                                    if response.status!=206 or response.headers.get('Content-Range')!=wanted:
                                        raise RuntimeError('invalid range response')
                                data=response.read(end-offset+2)
                                if len(data)!=end-offset+1:raise RuntimeError('range length mismatch')
                            break
                        except Exception:
                            if attempt==2:raise
                            time.sleep(2**attempt)
                    output.write(data);output.flush();offset=end+1
                    if offset==size or offset%(256*1024**2)==0:print('Progress:',name,offset,'/',size,flush=True)
            actual=digest(part)
            if expected and actual!=expected:raise RuntimeError('official LFS checksum mismatch')
            if not expected:
                blob=part.read_bytes()
                if hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()!=entry['blobId']:
                    raise RuntimeError('official Git blob mismatch')
            part.replace(path)
        actual=digest(path)
        if expected and actual!=expected:raise RuntimeError('existing model checksum mismatch')
        manifest['files'][name]={'bytes':size,'sha256':actual,'official_lfs_sha256':expected,'git_blob_id':entry['blobId']}
        (dest/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['completed']=True
    (dest/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Official FP32 layout downloaded and verified',flush=True)

if __name__=='__main__':main()
