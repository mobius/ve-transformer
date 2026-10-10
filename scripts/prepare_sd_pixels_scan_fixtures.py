"""Create float input fixtures; real decoded inputs come from CPU model reference."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 folder=ROOT/'build/sd-pixels-scan-probe/fixtures';folder.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(20261009);cases=[]
 edges=np.array([-np.finfo(np.float32).max,-1000,-2,-1,-0.,0.,1,2,1000,np.finfo(np.float32).max],dtype=np.float32)
 for spatial in (1,2,7,255,256,257,262143,262144):
  data=rng.uniform(-3,3,spatial*3).astype(np.float32);data[:min(len(data),len(edges))]=edges[:min(len(data),len(edges))]
  cases.append(dict(kind='random_edges_'+str(spatial),spatial=spatial,data=data))
 values=(np.arange(256,dtype=np.float32)+np.float32(.5))/np.float32(255)
 original=(values*np.float32(2)-np.float32(1)).astype(np.float32)
 data=np.concatenate([np.nextafter(original,np.float32(-np.inf)),original,np.nextafter(original,np.float32(np.inf))])
 cases.append(dict(kind='round_boundaries',spatial=256,data=data))
 for kind,value in (('zero',-1.),('white',1.),('alternating',0.)):
  data=np.full(262144*3,value,dtype=np.float32)
  if kind=='alternating':data[::2]=-1;data[1::2]=1
  cases.append(dict(kind=kind,spatial=262144,data=data))
 reference=ROOT/'build/results/20261008T081800Z-sd-reference';summary=reference/'summary.json';v=json.loads(summary.read_text())
 if not v['completed'] or v['size']!=512 or v['steps']!=1:raise RuntimeError('completed one-step reference required')
 for row in v['cases']:
  source=reference/('case'+str(row['case']))/'decoded.f32'
  if sha(source)!=row['arrays']['decoded']['sha256']:raise RuntimeError('real decoded input checksum differs')
  data=np.fromfile(source,dtype=np.float32)
  if data.size!=262144*3 or not np.isfinite(data).all():raise RuntimeError('real shape/finiteness differs')
  cases.append(dict(kind='cpu_real_'+str(row['case']),spatial=262144,data=data,source=str(source.relative_to(ROOT)),source_sha256=sha(source)))
 for name,value in (('nan',np.nan),('posinf',np.inf),('neginf',-np.inf)):
  for position in (0,385,770):
   data=np.linspace(-2,2,771,dtype=np.float32);data[position]=value
   cases.append(dict(kind=name+'_'+str(position),spatial=257,data=data,invalid_index=position))
 for spatial,positions in ((257,(0,385,770)),(257,(255,256,770)),(257,(256,257,770)),(257,(257,385,770)),(262144,(385,65535,786431)),(262144,(786431,))):
  data=np.linspace(-2,2,spatial*3,dtype=np.float32)
  for index,position in enumerate(positions):
   if index%3==0:data.view(np.uint32)[position]=np.uint32(0x7fc12345)
   elif index%3==1:data[position]=np.inf
   else:data[position]=-np.inf
  cases.append(dict(kind='nonfinite_boundaries_'+str(spatial)+'_'+str(positions[0]),spatial=spatial,data=data,invalid_index=positions[0],invalid_locations=list(positions)))
 records=[];lines=[]
 for index,item in enumerate(cases):
  data=item.pop('data');path=folder/('case%02d.f32'%index);path.write_bytes(data.tobytes());item.update(index=index,input_path=str(path.relative_to(ROOT)),input_sha256=sha(path),count=data.size)
  records.append(item);lines.append(str(index)+' '+str(item['spatial'])+' '+str(item.get('invalid_index',-1))+' '+str(path))
 (folder/'fixtures.tsv').write_text('\n'.join(lines)+'\n');(folder/'manifest.json').write_text(json.dumps(dict(cases=records,reference_summary=str(summary.relative_to(ROOT)),reference_summary_sha256=sha(summary),generator_sha256=sha(Path(__file__))),indent=2)+'\n');print('Pixel fixtures:',len(records))
if __name__=='__main__':main()
