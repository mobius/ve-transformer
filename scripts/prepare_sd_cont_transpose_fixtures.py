"""Bind actual top CONT geometries and immutable model baseline archives."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 proof=ROOT/'docs/results/20261009T183852Z-sd-turbo-cont-shapes.json';v=json.loads(proof.read_text());snapshot=ROOT/'build/accepted/cont-shapes-20261009T183852Z';manifest=snapshot/'build/sd-baseline-ve/manifest.json';current=ROOT/'build/sd-baseline-ve/manifest.json'
 if v['status']!='cont_shape_profile_verified' or sha(manifest)!=v['manifest_sha256'] or sha(current)!=v['manifest_sha256']:raise RuntimeError('accepted actual model required')
 for n,h in json.loads(manifest.read_text())['sha256'].items():
  if sha(snapshot/n)!=h or sha(ROOT/n)!=h:raise RuntimeError('baseline model input changed')
 index=json.loads((snapshot/'archive-sha256.json').read_text());libraries={}
 for n in ('ggml/src/libggml.a','ggml/src/ggml-cpu/libggml-cpu.a','ggml/src/libggml-base.a'):
  path='build/sd-baseline-ve/'+n
  if sha(snapshot/path)!=index[path] or sha(ROOT/path)!=index[path]:raise RuntimeError('actual baseline graph archive differs')
  libraries[path]=index[path]
 rows=[]
 for r in v['hot_shape_groups'][:3]:
  ne,nb=r['a'],r['an'];m=ne[0];n=ne[1]*ne[2]*ne[3]
  if r['stage']!='unet' or r['type']!=[0,0] or r['exact'] or r['overlap'] or not r['destination_contiguous'] or nb[0]!=4*n or (ne[1]>1 and nb[1]!=4) or (ne[2]>1 and nb[2]!=4*ne[1]) or (ne[3]>1 and nb[3]!=4*ne[1]*ne[2]):raise RuntimeError('actual transpose geometry differs')
  rows.append(dict(shape=ne,strides=nb,m=m,n=n,nodes=r['nodes'],diagnostic_seconds=r['seconds']))
 folder=ROOT/'build/sd-cont-transpose-probe';folder.mkdir(exist_ok=True)
 (folder/'shapes.tsv').write_text(''.join('\t'.join(map(str,[i]+r['shape']+r['strides']))+'\n' for i,r in enumerate(rows)))
 (folder/'baseline.json').write_text(json.dumps(dict(proof_sha256=sha(proof),manifest_sha256=sha(current),binary_sha256=v['binary_sha256'],actual_cpu_object_sha256=index['actual-ggml-cpu-object.o'],libraries_sha256=libraries,shapes=rows),indent=2)+'\n');print('Actual top-three CONT layouts and baseline archives bound')
if __name__=='__main__':main()
