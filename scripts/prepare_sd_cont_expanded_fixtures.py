"""Bind nine additional observed layouts to the accepted current model."""
import json
from pathlib import Path
from benchmark_sd_runtime import sha
ROOT=Path(__file__).resolve().parents[1]
def main():
 proof=ROOT/'docs/results/20261009T191953Z-sd-turbo-cont-abba.json'
 shape_proof=ROOT/'docs/results/20261009T183852Z-sd-turbo-cont-shapes.json'
 accepted=json.loads(proof.read_text());v=json.loads(shape_proof.read_text())
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 if accepted['status']!='formal_cont_abba_verified' or not accepted['all_trace_and_png_bytes_identical'] or sha(manifest)!=accepted['model_manifest_sha256']:raise RuntimeError('accepted current model required')
 for name,digest in json.loads(manifest.read_text())['sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('model inputs changed')
 if v['status']!='cont_shape_profile_verified' or len(v['hot_shape_groups'])!=48:raise RuntimeError('actual shape proof required')
 libraries={}
 index=json.loads((ROOT/'build/accepted/cont-model-20261009T185903Z/archive-sha256.json').read_text())
 for n in ('ggml/src/libggml.a','ggml/src/ggml-cpu/libggml-cpu.a','ggml/src/libggml-base.a'):
  name='build/sd-baseline-ve/'+n
  if sha(ROOT/name)!=index[name]:raise RuntimeError('accepted actual graph archive changed')
  libraries[name]=index[name]
 objects=list((ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir').rglob('sd-ggml-cpu.c.o'))
 if len(objects)!=1:raise RuntimeError('actual CPU backend object required')
 rows=[]
 for r in v['hot_shape_groups'][3:12]:
  ne,nb=r['a'],r['an'];m=ne[0];n=ne[1]*ne[2]*ne[3]
  if r['type']!=[0,0] or r['exact'] or r['overlap'] or not r['destination_contiguous'] or ne[3]!=1 or m>4096 or n>4096 or m*n>2097152 or nb[0]!=4*n or (ne[1]>1 and nb[1]!=4) or (ne[2]>1 and nb[2]!=4*ne[1]):raise RuntimeError('supported actual transpose geometry required')
  rows.append(dict(stage=r['stage'],shape=ne,strides=nb,m=m,n=n,nodes=r['nodes'],diagnostic_seconds=r['seconds']))
 folder=ROOT/'build/sd-cont-expanded-probe';folder.mkdir(exist_ok=True)
 (folder/'shapes.tsv').write_text(''.join('\t'.join(map(str,[i]+r['shape']+r['strides']))+'\n' for i,r in enumerate(rows)))
 (folder/'baseline.json').write_text(json.dumps(dict(proof_sha256=sha(proof),shape_proof_sha256=sha(shape_proof),manifest_sha256=sha(manifest),binary_sha256=accepted['binary_sha256'],actual_cpu_object_sha256=sha(objects[0]),libraries_sha256=libraries,shapes=rows),indent=2)+'\n')
 print('Nine additional actual CONT layouts and accepted current archives bound')
if __name__=='__main__':main()
