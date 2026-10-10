"""Prepare an isolated quiet diagnostic clone; accepted model is unchanged."""
import json,subprocess,sys
from pathlib import Path
from benchmark_sd_runtime import sha
folder=Path('build/sd-group-norm-batched-probe')
subprocess.run([sys.executable,'scripts/prepare_sd_group_norm_scale.py','--folder',str(folder)],check=True)
source=folder/'sd-ggml-cpu.c'
text=source.read_text()
needle='                fprintf(stderr,"SD_GGML_THREADS stage=%s actual=%d maximum=%d'+chr(92)+'n",'
if text.count(needle)!=1:raise RuntimeError('unique actual team diagnostic required')
text=text.replace(needle,'                const char * quiet=getenv("SD_GGML_THREADS_QUIET");'+chr(10)+'                if (!(quiet && strcmp(quiet,"1")==0))'+chr(10)+needle)
source.write_text(text)
baseline=folder/'baseline.json';data=json.loads(baseline.read_text());data['cloned_source_sha256']=sha(source);data['timing_scope']='diagnostics retained during correctness, suppressed during timed batches only; 16 calls per arm';baseline.write_text(json.dumps(data,indent=2)+'\n')
