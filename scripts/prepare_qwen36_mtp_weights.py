"""Fetch only native MTP/shared tensors from a pinned public checkpoint.

No authentication or global caches. Large tensor ranges are streamed to build/.
This establishes source provenance, not VE execution or target-version identity.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='Qwen/Qwen3.6-35B-A3B'
REVISION='995ad96eacd98c81ed38be0c5b274b04031597b0'
INDEX_SHA256='41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83'
FOLDER=ROOT/'build/models/qwen36-mtp-source'
PARTS=FOLDER/'tensor-parts'
BASE=f'https://huggingface.co/{REPO}/resolve/{REVISION}/'


def range_read(file,start,end,destination=None):
    # Keep each public HTTP request bounded: retrying a nearly completed GiB
    # transfer wastes bandwidth and exposes it to expiring signed redirects.
    if destination is not None:
        digest=hashlib.sha256()
        with destination.open('wb') as handle:
            for position in range(start,end+1,32*1024**2):
                stop=min(end,position+32*1024**2-1)
                block=range_read(file,position,stop)
                handle.write(block);digest.update(block)
                consumed=stop-start+1
                if consumed%(64*1024**2)==0 or stop==end:
                    print('range progress MiB',consumed//1024**2,'/',(end-start+1)//1024**2,flush=True)
        return digest.hexdigest()
    url=BASE+file+f'?download=true&range_start={start}&range_end={end}'
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=30) as response:
                expected=f'bytes {start}-{end}/'
                if response.status!=206 or not response.headers.get('Content-Range','').startswith(expected):
                    raise RuntimeError('range response mismatch; refusing full-file transfer')
                length=end-start+1
                data=[]
                consumed=0
                while consumed<length:
                    block=response.read(min(8*1024**2,length-consumed))
                    if not block:raise RuntimeError('truncated range response')
                    consumed+=len(block);data.append(block)
                return b''.join(data)
        except (OSError,RuntimeError) as error:
            print('range retry',file,'attempt',attempt+1,'error_type',type(error).__name__,flush=True)
            if attempt==2:raise RuntimeError('public tensor range download failed') from None
            time.sleep(1)


def main():
    FOLDER.mkdir(parents=True,exist_ok=True);PARTS.mkdir(exist_ok=True)
    index_path=ROOT/'build/results/qwen36-official-model.safetensors.index.json'
    if not index_path.exists() or hashlib.sha256(index_path.read_bytes()).hexdigest()!=INDEX_SHA256:
        with urllib.request.urlopen(BASE+'model.safetensors.index.json',timeout=30) as response:payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=INDEX_SHA256:raise RuntimeError('pinned index checksum mismatch')
        index_path.parent.mkdir(parents=True,exist_ok=True);index_path.write_bytes(payload)
    index=json.loads(index_path.read_text())['weight_map']
    keep={name for name in index if name.startswith('mtp.')}
    keep.update(('lm_head.weight','model.language_model.embed_tokens.weight','model.language_model.norm.weight'))
    assert len(keep)==22 and all(name in index for name in keep)
    headers={}
    for file in sorted({index[name] for name in keep}):
        size=struct.unpack('<Q',range_read(file,0,7))[0]
        assert 0<size<1024**2
        headers[file]=json.loads(range_read(file,8,7+size))
        headers[file]['_header_bytes']=8+size
    provenance={'repository':REPO,'revision':REVISION,'index_sha256':INDEX_SHA256,'completed':False,'tensors':[]}
    for i,name in enumerate(sorted(keep)):
        file=index[name];header=headers[file];spec=header[name]
        begin,end=spec['data_offsets'];length=end-begin
        part=PARTS/f'{i:03d}.bin';record=PARTS/f'{i:03d}.json'
        expected={'name':name,'file':file,'dtype':spec['dtype'],'shape':spec['shape'],'bytes':length,'source_offsets':[header['_header_bytes']+begin,header['_header_bytes']+end]}
        valid=False
        if part.exists() and record.exists() and part.stat().st_size==length:
            old=json.loads(record.read_text())
            if all(old.get(k)==v for k,v in expected.items()):
                h=hashlib.sha256()
                with part.open('rb') as handle:
                    for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
                valid=h.hexdigest()==old.get('sha256')
        if valid:expected['sha256']=old['sha256']
        else:
            print('tensor',i,name,'MiB',round(length/1024**2,2),flush=True)
            expected['sha256']=range_read(file,header['_header_bytes']+begin,header['_header_bytes']+end-1,part)
            record.write_text(json.dumps(expected,indent=2)+'\n')
        provenance['tensors'].append(expected)
        (FOLDER/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    output_header={};offset=0
    for tensor in provenance['tensors']:
        output_header[tensor['name']]={'dtype':tensor['dtype'],'shape':tensor['shape'],'data_offsets':[offset,offset+tensor['bytes']]}
        offset+=tensor['bytes']
    encoded=json.dumps(output_header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    output=FOLDER/'model.safetensors';temporary=FOLDER/'model.safetensors.partial'
    with temporary.open('wb') as handle:
        handle.write(struct.pack('<Q',len(encoded)));handle.write(encoded)
        for i in range(len(provenance['tensors'])):
            with (PARTS/f'{i:03d}.bin').open('rb') as source:shutil.copyfileobj(source,handle,8*1024**2)
    temporary.replace(output)
    for file in ('config.json','tokenizer.json','tokenizer_config.json'):
        with urllib.request.urlopen(BASE+file,timeout=30) as response:(FOLDER/file).write_bytes(response.read())
    auxiliary={file:hashlib.sha256((FOLDER/file).read_bytes()).hexdigest() for file in ('config.json','tokenizer.json','tokenizer_config.json')}
    provenance.update(completed=True,payload_bytes=offset,assembled_file='model.safetensors',auxiliary_sha256=auxiliary)
    (FOLDER/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('MTP source ready, selected payload GiB',round(offset/1024**3,3),flush=True)


if __name__=='__main__':main()
