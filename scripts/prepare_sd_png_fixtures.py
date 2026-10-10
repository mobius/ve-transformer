"""Create boundary and accepted real RGB fixtures without model recomputation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 folder=ROOT/'build/sd-png-probe/fixtures';folder.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(20261009);cases=[]
 for width,height,kind in ((1,1,'zero'),(2,3,'checker'),(17,19,'gradient'),(255,257,'random'),(512,512,'zero'),(512,512,'white'),(512,512,'gradient'),(512,512,'checker'),(512,512,'random')):
  yy,xx=np.indices((height,width));data=np.empty((height,width,3),dtype=np.uint8)
  if kind=='random':data=rng.integers(0,256,size=data.shape,dtype=np.uint8)
  elif kind=='zero':data.fill(0)
  elif kind=='white':data.fill(255)
  elif kind=='gradient':
   for channel in range(3):data[:,:,channel]=(xx*17+yy*31+channel*79)%256
  else:
   for channel in range(3):data[:,:,channel]=((xx+yy+channel)%2)*255
  cases.append(dict(width=width,height=height,kind=kind,data=data))
 proof_path=ROOT/'docs/results/20261009T143607Z-sd-turbo-gemm-spatial-model.json';proof=json.loads(proof_path.read_text());test=next(t for t in proof['tests'] if t['name']=='six');native_path=ROOT/test['artifacts']/'summary.json'
 if sha(native_path)!=test['sha256'][str(native_path.relative_to(ROOT))]:raise RuntimeError('accepted real image summary changed')
 native=json.loads(native_path.read_text())
 for index,request in enumerate(native['requests']):
  png=ROOT/test['artifacts']/('request'+str(index))/'image.png'
  if sha(png)!=request['png_sha256']:raise RuntimeError('accepted real PNG changed')
  with Image.open(png) as image:data=np.asarray(image.convert('RGB')).copy()
  if data.shape!=(512,512,3):raise RuntimeError('matching image dimensions required')
  cases.append(dict(width=512,height=512,kind='real'+str(index),data=data,source_png=str(png.relative_to(ROOT)),source_png_sha256=sha(png)))
 lines=[];records=[]
 for index,case in enumerate(cases):
  data=case.pop('data');path=folder/('case%02d.rgb'%index);path.write_bytes(data.tobytes());record=dict(index=index,**case,rgb_path=str(path.relative_to(ROOT)),rgb_sha256=sha(path),rgb_bytes=data.size);records.append(record)
  lines.append(str(index)+' '+str(case['width'])+' '+str(case['height'])+' '+str(path))
 (folder/'fixtures.tsv').write_text('\n'.join(lines)+'\n');(folder/'manifest.json').write_text(json.dumps(dict(cases=records,source_proof=str(proof_path.relative_to(ROOT)),source_proof_sha256=sha(proof_path),generator_sha256=sha(Path(__file__))),indent=2)+'\n');print('PNG fixtures: 9 boundary/content plus 6 accepted real RGB')
if __name__=='__main__':main()
