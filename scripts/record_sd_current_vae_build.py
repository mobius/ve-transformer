"""Bind a one-clause dispatch extension to unchanged numerical objects."""
import hashlib,json,subprocess
from pathlib import Path
from prepare_sd_current_vae_gate import instrument
ROOT=Path(__file__).resolve().parents[1]
BEFORE=ROOT/'build/diagnostics/current-vae-before-20261010T002648Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def evidence():
 index=json.loads((BEFORE/'archive-sha256.json').read_text())
 for name,digest in index.items():
  if sha(BEFORE/name)!=digest:raise RuntimeError('frozen before-build evidence changed: '+name)
 prior=json.loads((ROOT/'docs/results/20261010T000834Z-sd-turbo-gemm-untiled-abba.json').read_text())
 manifest=ROOT/'build/sd-baseline-ve/manifest.json'
 old=json.loads((BEFORE/'build/sd-baseline-ve/manifest.json').read_text())
 new=json.loads(manifest.read_text())
 if sha(BEFORE/'build/sd-baseline-ve/manifest.json')!=prior['model_manifest_sha256']:raise RuntimeError('accepted manifest binding differs')
 if sha(BEFORE/'build/sd-baseline-ve/bin/sd')!=prior['binary_sha256']:raise RuntimeError('accepted binary binding differs')
 cpu='build/sd-baseline-overlay/sd-ggml-cpu.c'
 if (ROOT/cpu).read_text()!=instrument((BEFORE/cpu).read_text()):raise RuntimeError('generated backend differs beyond exact thread gate')
 expected={'scripts/prepare_sd_baseline_overlay.py','scripts/record_sd_baseline_build.py',cpu,'build/sd-baseline-ve/bin/sd'}
 changed={n for n,d in old['sha256'].items() if new['sha256'].get(n)!=d}
 if changed!=expected or set(new['sha256'])-set(old['sha256'])!={'scripts/prepare_sd_current_vae_gate.py'}:raise RuntimeError('unexpected manifest input delta')
 for n,d in new['sha256'].items():
  if sha(ROOT/n)!=d:raise RuntimeError('manifest input changed: '+n)
 objects=[n for n in index if n.endswith(('.o','.a'))]
 changed_objects=[n for n in objects if sha(ROOT/n)!=index[n]]
 cpu_objects=[n for n in changed_objects if n.endswith('/sd-ggml-cpu.c.o')]
 archives=[n for n in changed_objects if n.endswith('/libggml-cpu.a')]
 if len(changed_objects)!=2 or len(cpu_objects)!=1 or len(archives)!=1:raise RuntimeError('only backend object and its archive may change')
 obj=ROOT/cpu_objects[0];archive=ROOT/archives[0]
 if obj.stat().st_mtime_ns<(ROOT/cpu).stat().st_mtime_ns:raise RuntimeError('backend object predates source')
 member=subprocess.check_output(['ar','p',str(archive),'sd-ggml-cpu.c.o'])
 if hashlib.sha256(member).hexdigest()!=sha(obj):raise RuntimeError('archive backend member differs')
 probe=ROOT/'build/results/20261010T003711Z-im2col-current4'
 log=ROOT/'build/im2col-current4.log';text=log.read_text()
 lines=[s for s in text.splitlines() if s.startswith('IM2COL_CURRENT4 width=')]
 if len(lines)!=8 or any('threads=4 ' not in s or 'input_unchanged=1 guards=1 PASS' not in s for s in lines) or 'IM2COL_CURRENT4_COMPLETE shapes=4 modes=2 actual_threads=4' not in text:raise RuntimeError('complete actual four-thread full-output proof required')
 if sum(int(s.split('checked_elements=')[1].split()[0]) for s in lines)!=2717908992:raise RuntimeError('full-output element count differs')
 for leaf in ('ve_sd_turbo_im2col.c.o','ve_sd_turbo_im2col_rows.c.o'):
  actual=[n for n in objects if n.endswith('/'+leaf)]
  if len(actual)!=1 or sha(probe/leaf)!=sha(ROOT/actual[0]):raise RuntimeError('probe differs from actual unchanged object')
 from record_qwen36_mtp import thermal
 from record_sd_im2col_result import fan_detail
 return dict(scope='exact four-or-eight thread dispatch gate; mathematical objects unchanged; complete model validation still required',before_index_sha256=sha(BEFORE/'archive-sha256.json'),manifest_sha256=sha(manifest),binary_sha256=sha(ROOT/'build/sd-baseline-ve/bin/sd'),checker_sha256=sha(ROOT/'tests/check_sd_resident.py'),changed_manifest_inputs=sorted(changed),changed_objects=[n.replace(str(ROOT).lstrip('/')+'/', 'project/') for n in changed_objects],unchanged_object_count=len(objects)-2,object_sha256={n.replace(str(ROOT).lstrip('/')+'/', 'project/'):sha(ROOT/n) for n in objects},probe_artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in probe.iterdir() if p.is_file()},probe_log_sha256=sha(log),probe_checked_elements=2717908992,probe_temperature=thermal(log),probe_fan_detail=fan_detail(log),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
 value=evidence();from record_qwen36_mtp import safe;safe(value);out=ROOT/'docs/results/20261010T002648Z-sd-turbo-current-vae-build.json';out.write_text(json.dumps(value,indent=2)+'\n');print('Current VAE gate build bound; unchanged objects:',value['unchanged_object_count'])

def retained_compilers():
 """Retain historical mathematics proofs only after exact current delta audit."""
 current=evidence()
 prior_path=ROOT/'docs/results/20261010T000834Z-sd-turbo-gemm-untiled-abba.json'
 prior=json.loads(prior_path.read_text())
 result={}
 for key in ('actual_spatial_compiler','actual_cont_compiler','actual_png_compiler','actual_softmax_scale_compiler','actual_group_norm_compiler'):
  result[key]=dict(scope='historical compiler and independent mathematics proof, retained via exact current dispatch-only source and object audit',historical_proof_sha256=sha(prior_path),historical_evidence=prior[key],current_build_delta=current)
 return result
