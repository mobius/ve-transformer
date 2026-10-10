#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -eq 1 && $1 == docs/results/*.json ]] || exit 2
pixel_model_proof=$1
pixel_model_name=four
pixel_model_reference=build/results/20261008T083810Z-sd-reference
pixel_model_cases=0,0
pixel_model_log=build/sd-gemm-spatial-small-model-four.log
 .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 2 -- \
  env VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 SD_VE_GROUP_NORM=1 SD_GROUP_NORM_SHAPE_PROFILE=0 SD_VE_SOFTMAX_SCALE=1 SD_SOFTMAX_SHAPE_PROFILE=0 SD_VE_CONT=extended SD_CONT_SHAPE_PROFILE=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash scripts/run_sd_gemm_small_four_after_cooling.sh \
  -- .venv/bin/python tests/check_sd_resident.py --reference "$pixel_model_reference" --mode resident \
  --cases "$pixel_model_cases" --tokenizer resident --nlc-threads unified --binary-scalar ve \
  --vae-blas-threads 4 --gelu ve --im2col-mode rows_256 --vae-spatial-tile mixed4096 \
  --png-encoder ve --rgb-buffer resident --pixel-kernel ve > "$pixel_model_log" 2>&1
 read -r pixel_model_run pixel_model_memory < <(.venv/bin/python -c 'import re,sys; from pathlib import Path; s=Path(sys.argv[1]).read_text(); r=re.findall(r"Native sequential requests verified: (build/results/[^\s]+)",s); m=re.findall(r"VE memory samples: (build/results/[^\s]+)",s); assert len(r)==len(m)==1; print(r[0],m[0])' "$pixel_model_log")
 .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 2 -- \
  env VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 SD_VE_GROUP_NORM=1 SD_GROUP_NORM_SHAPE_PROFILE=0 SD_VE_SOFTMAX_SCALE=1 SD_SOFTMAX_SHAPE_PROFILE=0 SD_VE_CONT=extended SD_CONT_SHAPE_PROFILE=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/cpu_duty.py --percent 25 -- taskset -c 0-3 \
  build/venvs/sd-image/bin/python scripts/record_sd_gemm_spatial_small_model.py --proof "$pixel_model_proof" \
  --name "$pixel_model_name" --run "$pixel_model_run" --memory "$pixel_model_memory" --guard-log "$pixel_model_log" \
  > "build/sd-gemm-spatial-small-model-${pixel_model_name}-audit.log" 2>&1
printf 'Mixed spatial model four retry: VE and guarded CPU audits completed\n'
