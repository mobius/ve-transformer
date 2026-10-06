"""Isolated numerical-precision experiment; run through temperature_guard.py."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/vendor/llama.cpp'
OUT=ROOT/'build/qwen-ve-precision'
STRICT=['-fno-fast-math','-fno-associative-math','-fno-reciprocal-math',
        '-mvector-sqrt-instruction','-mvector-floating-divide-instruction',
        '-fdiag-inline=0','-fdiag-vector=0']

def execute(command):
    subprocess.run(command,cwd=ROOT,check=True)

def settings(target,language):
    makefile=ROOT/'build/llama-ve/ggml/src/CMakeFiles'/('ggml-'+target+'.dir')/'flags.make'
    values={}
    for line in makefile.read_text().splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1);values[key]=shlex.split(value)
    return values[language+'_DEFINES']+values[language+'_INCLUDES']+values[language+'_FLAGS']+STRICT

def main():
    execute(['bash','scripts/check_environment.sh'])
    revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if revision!='8d81559fa7b8bcac9f7c8b478858953486371f90':raise RuntimeError('unexpected framework revision')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain']):raise RuntimeError('modified framework')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('libggml-cpu.a','libggml-base.a'):
        shutil.copyfile(ROOT/'build/llama-ve/ggml/src'/name,OUT/name)
    manifest=dict(revision=revision,precision_flags=STRICT,objects=[])
    sources=[('base','ggml-quants.c'),('cpu','ggml-cpu/quants.c'),
             ('cpu','ggml-cpu/vec.cpp'),('cpu','ggml-cpu/binary-ops.cpp'),
             ('cpu','ggml-cpu/unary-ops.cpp'),('cpu','ggml-cpu/ops.cpp')]
    for target,relative in sources:
        source=SOURCE/'ggml/src'/relative
        member=source.name+'.o'
        directory=OUT/target;directory.mkdir(exist_ok=True)
        output=directory/member
        language='C' if source.suffix=='.c' else 'CXX'
        # NCC 5.4.1 exhausts its internal optimizer pool for strict O3 ops.cpp.
        extra=['-O1','-fno-inline'] if relative=='ggml-cpu/ops.cpp' else []
        print('Compiling precision object '+target+'/'+member,flush=True)
        execute(['taskset','-c','0-23','ncc' if language=='C' else 'nc++',
                 *settings(target,language),*extra,'-c',str(source),'-o',str(output)])
        archive=OUT/('libggml-'+target+'.a')
        members=subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()
        if members.count(member)!=1:raise RuntimeError('archive member identity mismatch')
        execute(['ar','r',str(archive),str(output)])
        manifest['objects'].append(dict(source=relative,archive=archive.name,member=member,extra_flags=extra,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            object_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    inc=['-I'+str(SOURCE/'include'),'-I'+str(SOURCE/'ggml/include'),'-I'+str(SOURCE/'ggml/src')]
    execute(['taskset','-c','0-23','ncc','-O3','-std=c11',*STRICT,*inc,
             '-c','src/ve_quant_kernels.c','-o',str(OUT/'ve_quant_kernels.o')])
    execute(['taskset','-c','0-23','nc++','-O3','-std=c++17',*STRICT,*inc,
             '-I'+str(SOURCE/'ggml/src/ggml-cpu'),'-I/opt/nec/ve/nlc/3.1.0/include','-DQWEN_NLC',
             '-c','src/qwen_nlc_hook.cpp','-o',str(OUT/'qwen_nlc_hook.o')])
    common=['taskset','-c','0-23','nc++','-O3','-std=c++17',
            '-Wl,-rpath-link,/opt/nec/ve/ncc/5.4.1/lib','build/qwen_infer.o','build/nec_llama_compat.o']
    libraries=['build/llama-ve/src/libllama.a','build/llama-ve/ggml/src/libggml.a',
               str(OUT/'libggml-cpu.a'),str(OUT/'libggml-base.a')]
    execute([*common,*libraries,'-lpthread','-ldl','-lm','-o',str(OUT/'qwen-infer-ve-baseline')])
    wrap=['-Wl,--wrap='+name for name in ('ggml_vec_dot_q4_K_q8_K','ggml_vec_dot_q5_K_q8_K',
                                         'ggml_vec_dot_q6_K_q8_K','ggml_vec_dot_q8_0_q8_0')]
    execute([*common,str(OUT/'ve_quant_kernels.o'),'build/ve_quant_wrap.o',*wrap,*libraries,
             '-lpthread','-ldl','-lm','-o',str(OUT/'qwen-infer-ve')])
    execute([*common,'-fopenmp',str(OUT/'ve_quant_kernels.o'),str(OUT/'qwen_nlc_hook.o'),
             '-Wl,--wrap=ggml_cpu_extra_compute_forward',*libraries,'-L/opt/nec/ve/nlc/3.1.0/lib',
             '-lcblas','-lblas_openmp','-lpthread','-ldl','-lm','-o',str(OUT/'qwen-infer-ve-nlc')])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Isolated precision executors linked; no hardware numerical claim yet.',flush=True)

if __name__=='__main__':main()
