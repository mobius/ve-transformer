"""Attach completed CPU-validation temperature evidence without redoing tests."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from benchmark_sd_runtime import sha
from record_qwen36_mtp import safe,thermal
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();result=a.result.resolve();result.relative_to(ROOT/'docs/results');log=a.log.resolve();log.relative_to(ROOT/'build');text=log.read_text()
 if ('Formal PNG ABBA audited: '+str(result.relative_to(ROOT))) not in text or 'Temperature summary:' not in text or 'Temperature guard stopped command' in text:raise RuntimeError('successful guarded CPU-validation completion required')
 paths=re.findall(r'log=(build/results/[^\s]+\.csv)',text)
 if len(paths)!=2:raise RuntimeError('validation thermal/fan CSVs required')
 rows=list(csv.DictReader((ROOT/paths[0]).open()))
 if not rows or any(not math.isfinite(float(r['temperature_c'])) or float(r['temperature_c'])>=float(r['stop_c']) for r in rows):raise RuntimeError('unsafe CPU-validation temperature')
 value=json.loads(result.read_text())
 if value['status']!='formal_abba_verified' or value['independent_cpu_checks']!=142 or not value['all_trace_and_png_bytes_identical']:raise RuntimeError('complete formal PNG checks required')
 for name,digest in value['artifact_sha256'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('encoder evidence changed')
 value['cpu_validation_temperature']=thermal(log);value['temperature_append_publisher_sha256']=sha(Path(__file__));value['artifact_sha256'].update({str(log.relative_to(ROOT)):sha(log)}|{name:sha(ROOT/name) for name in paths})
 flags=ROOT/'build/sd-baseline-ve/examples/cli/CMakeFiles/sd.dir/flags.make';lines=re.findall(r'^CXX_FLAGS = (.*)$',flags.read_text(),re.M)
 if len(lines)!=1:raise RuntimeError('actual main compiler flags required')
 opts=re.findall(r'(?<!\S)-O[0-3sg](?!\S)',lines[0]);value['current_model_main_compiler_flags']=lines[0];value['current_model_main_optimization']=opts[-1];value['artifact_sha256'][str(flags.relative_to(ROOT))]=sha(flags)
 safe(value);result.write_text(json.dumps(value,indent=2)+'\n');print('Guarded CPU validation temperature appended; actual model main optimization:',opts[-1])
if __name__=='__main__':main()
