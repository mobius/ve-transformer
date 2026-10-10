"""Bind optional packing integration to exact sources, real objects and bit tests."""
import hashlib,json,re,subprocess
from pathlib import Path
from prepare_sd_gemm_pretranspose_model import instrument
from record_qwen36_mtp import safe,thermal
from record_sd_im2col_result import fan_detail
ROOT=Path(__file__).resolve().parents[1]
BEFORE=ROOT/'build/diagnostics/pretranspose-model-before-20261010T012844Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def functions(p):
 result=[]
 for row in subprocess.check_output(['readelf','-Ws',str(p)],text=True).splitlines():
  parts=row.split()
  if len(parts)>=8 and parts[3]=='FUNC' and parts[7]=='sd_ve_gemm_pack_f32' and int(parts[2])>0:result.append((parts[7],int(parts[2])))
 return result
def evidence():
 index=json.loads((BEFORE/'archive-sha256.json').read_text())
 for n,d in index.items():
  if sha(BEFORE/n)!=d:raise RuntimeError('before snapshot changed')
 previous=ROOT/'docs/results/20261010T005555Z-sd-turbo-current-vae-abba.json';prior=json.loads(previous.read_text())
 old=json.loads((BEFORE/'build/sd-baseline-ve/manifest.json').read_text());manifest=ROOT/'build/sd-baseline-ve/manifest.json';new=json.loads(manifest.read_text())
 if sha(BEFORE/'build/sd-baseline-ve/manifest.json')!=prior['model_manifest_sha256'] or sha(BEFORE/'build/sd-baseline-ve/bin/sd')!=prior['binary_sha256'] or sha(BEFORE/'tests/check_sd_resident.py')!=prior['checker_sha256']:raise RuntimeError('accepted before model binding differs')
 source=ROOT/'build/sd-baseline-overlay/sd-ggml-blas.cpp'
 if source.read_text()!=instrument((BEFORE/'build/sd-baseline-overlay/sd-ggml-blas.cpp').read_text()):raise RuntimeError('generated BLAS differs beyond exact optional integration')
 expected={'scripts/prepare_sd_baseline_overlay.py','scripts/record_sd_baseline_build.py','cmake/nec-sd-baseline-overrides.cmake','build/sd-baseline-overlay/sd-ggml-blas.cpp','build/sd-baseline-ve/bin/sd'}
 changed={n for n,d in old['sha256'].items() if new['sha256'].get(n)!=d}
 if changed!=expected or set(new['sha256'])-set(old['sha256'])!={'scripts/prepare_sd_gemm_pretranspose_model.py','src/ve_sd_turbo_gemm_pack.c'}:raise RuntimeError('unexpected model input delta')
 for n,d in new['sha256'].items():
  if sha(ROOT/n)!=d:raise RuntimeError('manifest input changed')
 objects=[n for n in index if n.endswith(('.o','.a'))];different=[n for n in objects if sha(ROOT/n)!=index[n]]
 if sorted(Path(n).name for n in different)!=['libggml-blas.a','sd-ggml-blas.cpp.o']:raise RuntimeError('unrelated mathematical object changed')
 folder=ROOT/'build/sd-baseline-ve/ggml/src/ggml-blas/CMakeFiles/ggml-blas.dir';packs=list(folder.rglob('ve_sd_turbo_gemm_pack.c.o'))
 if len(packs)!=1:raise RuntimeError('unique actual pack object required')
 obj=packs[0];archive=ROOT/'build/sd-baseline-ve/ggml/src/ggml-blas/libggml-blas.a';flags=folder/'flags.make';make=folder/'build.make'
 if sorted(subprocess.check_output(['ar','t',str(archive)],text=True).splitlines())!=['sd-ggml-blas.cpp.o','ve_sd_turbo_gemm_pack.c.o']:raise RuntimeError('exact actual archive members required')
 if hashlib.sha256(subprocess.check_output(['ar','p',str(archive),'ve_sd_turbo_gemm_pack.c.o'])).hexdigest()!=sha(obj):raise RuntimeError('archive pack member differs')
 if obj.stat().st_mtime_ns<(ROOT/'src/ve_sd_turbo_gemm_pack.c').stat().st_mtime_ns:raise RuntimeError('pack object stale')
 commands=[s for s in make.read_text().splitlines() if '$(C_FLAGS)' in s and ' -c ' in s and 've_sd_turbo_gemm_pack.c' in s]
 base=re.findall(r'^C_FLAGS = (.*)$',flags.read_text(),re.M)
 if len(commands)!=1 or len(base)!=1:raise RuntimeError('unique actual pack command required')
 effective=base[0]+' '+commands[0]
 if re.findall(r'(?<!\S)-O[0-3sg](?!\S)',effective)[-1:]!=['-O2'] or any(s not in effective for s in ('-fno-fast-math','-fno-associative-math','-fopenmp')):raise RuntimeError('strict actual O2 pack flags required')
 sizes=functions(obj)
 if len(sizes)!=1 or sizes!=functions(ROOT/'build/sd-baseline-ve/bin/sd'):raise RuntimeError('actual linked pack function differs')
 log=ROOT/'build/sd-gemm-pack-actual.log';text=log.read_text()
 paths=re.findall(r'Actual pack kernel artifacts: (build/results/[^\s]+)',text)
 if len(paths)!=1 or text.count('PACK_COMPLETE shapes=3 modes=2 actual_threads=4 rejected=10')!=1:raise RuntimeError('completed actual-object bit suite required')
 run=ROOT/paths[0]
 rows=re.findall(r'PACK_CHECK shape=(\d+) n=(\d+) k=(\d+) mode=(\d+) seconds=([0-9.]+) checked=(\d+) input_unchanged=1 guards=1 PASS',text)
 expected=[(i,n,k,mode,n*k) for i,(n,k) in enumerate(((2048,4608),(4096,2304),(4096,4608))) for mode in range(2)]
 if len(rows)!=6 or [(int(r[0]),int(r[1]),int(r[2]),int(r[3]),int(r[5])) for r in rows]!=expected or sum(int(r[5]) for r in rows)!=75497472:raise RuntimeError('complete actual bit checks required')
 if sha(run/'pack.o')!=sha(obj) or sha(run/'ve_sd_turbo_gemm_pack.c')!=sha(ROOT/'src/ve_sd_turbo_gemm_pack.c'):raise RuntimeError('actual test and model packing object differ')
 probe=ROOT/'docs/results/20261010T012321Z-sd-turbo-gemm-pretranspose.json';measured=json.loads(probe.read_text())
 if not measured.get('cpu_validation_temperature') or measured['full_blas_comparisons']!=48 or any(not r['both_pairs_positive'] for r in measured['comparisons']):raise RuntimeError('finalized preparation-inclusive matrix evidence required')
 libraries={n:sha(Path('/opt/nec/ve/nlc/3.1.0/lib')/n) for n in ('libcblas.so','libblas_openmp.so')}
 if libraries!=measured['nlc_library_sha256']:raise RuntimeError('matrix library differs from probe')
 result=dict(before_index_sha256=sha(BEFORE/'archive-sha256.json'),accepted_before_proof_sha256=sha(previous),independent_cost_proof_sha256=sha(probe),model_manifest_sha256=sha(manifest),binary_sha256=sha(ROOT/'build/sd-baseline-ve/bin/sd'),checker_sha256=sha(ROOT/'tests/check_sd_resident.py'),changed_model_inputs=sorted(changed),changed_existing_objects=sorted(Path(n).name for n in different),unchanged_existing_objects=len(objects)-2,pack_source_sha256=sha(ROOT/'src/ve_sd_turbo_gemm_pack.c'),pack_object_sha256=sha(obj),pack_archive_sha256=sha(archive),pack_flags_make_sha256=sha(flags),pack_build_make_sha256=sha(make),pack_linked_function_size=sizes[0][1],compiler_optimization='O2 strict floating point and OpenMP, data-only pack',actual_bit_checked_elements=75497472,actual_bit_timings_seconds=[float(r[4]) for r in rows],actual_bit_temperature=thermal(log),actual_bit_fan_detail=fan_detail(log),actual_bit_artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [log]+list(run.iterdir()) if p.is_file()},nlc_library_sha256=libraries,scope='exact optional three-shape packing path; new data-only object and BLAS caller/archive; all other mathematical objects unchanged; default off; model numerical and image verification pending',verifier_sha256=sha(Path(__file__)))
 safe(result);return result
if __name__=='__main__':
 v=evidence();out=ROOT/'docs/results/20261010T012844Z-sd-turbo-gemm-pretranspose-build.json';out.write_text(json.dumps(v,indent=2)+'\n');print('Optional pretranspose actual build verified:',v['unchanged_existing_objects'],'unchanged objects/archives')
