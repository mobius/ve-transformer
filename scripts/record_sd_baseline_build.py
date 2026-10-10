"""Capture the compact native image build and its exact adaptations."""
import hashlib,json,os,subprocess,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    source=ROOT/'build/vendor/sd-turbo-baseline'
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    ggml_revision=subprocess.check_output(['git','-C',str(source/'ggml'),'rev-parse','HEAD'],text=True).strip()
    if revision!='5eb15ef4d022bef4a391de4f5f6556e81fbb5024' or ggml_revision!='6fcbd60bc72ac3f7ad43f78c87e535f2e6206f58':raise RuntimeError('unexpected compact source revision')
    for tree in (source,source/'ggml'):
        if subprocess.check_output(['git','-C',str(tree),'status','--porcelain']):raise RuntimeError('vendor source modified')
    files=[ROOT/x for x in ('scripts/build_sd_baseline.sh','scripts/record_sd_baseline_build.py','scripts/prepare_sd_baseline_source.sh',
          'scripts/prepare_sd_baseline_overlay.py','scripts/prepare_sd_gemm_spatial_small_model.py','scripts/prepare_sd_gemm_untiled_model.py','scripts/prepare_sd_gemm_pretranspose_model.py','scripts/prepare_sd_current_vae_gate.py','scripts/prepare_sd_group_norm_profile.py','scripts/prepare_sd_group_norm_model.py','scripts/prepare_sd_weight_index.py','cmake/nec-sd-baseline-overrides.cmake','cmake/nec-sd-baseline-toolchain.cmake','cmake/nec-ve.cmake',
          'build/sd-baseline-ve/bin/sd','build/vendor/sd-turbo-baseline/thirdparty/stb_image_write.h')]
    files+=sorted((ROOT/'build/sd-baseline-overlay').glob('*'))
    files+=sorted((ROOT/'src').glob('ve_sd_turbo_*'))
    files+=sorted((ROOT/'build/models/sd-turbo').rglob('*.ve-index'))
    files=[p for p in files if p.is_file()]
    # NCC's generated depfiles omit included headers; CMake supplies explicit
    # dependencies. Reject a manifest if component objects predate these inputs.
    headers=list((ROOT/'build/sd-baseline-overlay').glob('*.h'))+list((ROOT/'build/sd-baseline-overlay').glob('*.hpp'))
    headers+=list((ROOT/'src').glob('ve_sd_turbo_*.h'))
    latest_header=max(p.stat().st_mtime_ns for p in headers)
    for component in ('clip','unet','vae'):
        path=ROOT/'src'/('ve_sd_turbo_'+component+'.cpp')
        obj=ROOT/'build/sd-baseline-ve/CMakeFiles/stable-diffusion.dir'/(str(path).lstrip('/')+'.o')
        if not obj.exists() or obj.stat().st_mtime_ns<max(latest_header,path.stat().st_mtime_ns):
            raise RuntimeError('component object predates source or explicit headers')
    dynamic=subprocess.check_output(['readelf','-d',str(ROOT/'build/sd-baseline-ve/bin/sd')],text=True)
    needed=re.findall(r'\(NEEDED\).*?\[([^]]+)\]',dynamic)
    mode=os.environ.get('SD_NLC_MODE','sequential')
    unified=os.environ.get('SD_GGML_OPENMP','0')=='1'
    flags=(ROOT/'build/sd-baseline-ve/ggml/src/ggml-cpu/CMakeFiles/ggml-cpu.dir/flags.make').read_text()
    if ('-DGGML_USE_OPENMP' in flags) != unified or ('-fopenmp' in flags) != unified:
        raise RuntimeError('actual generic backend compile flags differ from requested runtime')
    if unified and mode!='openmp':
        raise RuntimeError('unified runtime requires OpenMP BLAS')
    expected_library='libblas_openmp' if mode=='openmp' else 'libblas_sequential'
    wrong_library='libblas_sequential' if mode=='openmp' else 'libblas_openmp'
    if not any(name.startswith(expected_library) for name in needed) or any(name.startswith(wrong_library) for name in needed):
        raise RuntimeError('actual NLC dependency differs from requested build')
    report={'backend':'ve','framework_variant':'compact','framework_revision':revision,'ggml_revision':ggml_revision,
            'compiler_standard':'C++11 image library; existing C++17 ggml backend',
            'precision':'FP32 weights, convolution packing and erf GELU','compiler_optimization':'numerical kernels O1; optional isolated GELU erf, SiLU and Softmax exp O2 with checked vector math; Softmax sum O1 without vectorization/reassociation; optional data-only im2col O2; model metadata and graph orchestration O0 without inlining; no fast-math',
            'compiler_duty_percent':int(os.environ.get('SD_BUILD_DUTY_PERCENT','50')),
            'header_dependency_tracking':'explicit CMake object dependencies; component freshness checked',
            'nlc_mode':mode,'dynamic_dependencies':needed,
            'generic_thread_runtime':'openmp' if unified else 'pthread',
            'nlc_thread_lifecycle':('explicit fixed eight mode supported with unified generic OpenMP' if unified else 'per BLAS graph thread selection, actual diagnostic region, reset to one before generic work') if mode=='openmp' else 'sequential library',
            'blas':'NEC NLC '+os.environ.get('SD_NLC_MODE','sequential')+', graph scheduler with native VE generic fallback',
            'validation_scope':'official SD-Turbo only; split CLIP/UNet/VAE with native EPS Euler and trailing timesteps; exported sigmas supplied by checker',
            'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    out=ROOT/'build/sd-baseline-ve/manifest.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print('Compact image manifest captured:',out.relative_to(ROOT))
if __name__=='__main__':main()
