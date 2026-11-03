#!/bin/bash
# Study 12 - NVIDIA DRIVE AGX Thor. The on-device steps (latency, quantized PyTorch probes, TensorRT engines for the
# action expert, compiled VLM, Qwen-Drive fidelity) are documented with exact commands in deploy/thor/README.md; copy
# the device outputs back to $FROST_ARTIFACTS under the names below. This script runs the workstation analyses.
# Protocol: protocols/thor.md. Released outputs: results/thor_v1_analysis_p0p1p2_v1_20261031,
# results/thor_trt_frost400_analysis_v0v1v2_v1_20261031, results/thor_trt_precision_flips_v1_20261031,
# results/thor_addendum3_analysis_v1_20261101 and the device summaries results/thor_{v1_20261030,trt_bench_20261031,
# vlm_trt_20261031,qwendrive_20261101}.
source "$(dirname "$0")/common.sh"
if stage analysis; then
  log "Protocol v1: P0 (bf16) hardware check against the workstation, P1/P2 quantized variants, 60 clips"
  $PY_ANALYSIS -m frost.analysis.thor_analysis --thor $A/thor_v1_20261030 --range 0 60 --out $A/thor_v1_analysis_p0p1p2_v1_20261031
  log "Addendum 2: TensorRT engines (bf16, FP8, NVFP4) on 400 clips; same estimator, bootstrap seed 20261031"
  T=$A/thor_trt_frost400_20261031; mkdir -p $T/as_v1
  ln -sfn ../trt_bf16 $T/as_v1/probe_bf16; ln -sfn ../trt_fp8 $T/as_v1/P1; ln -sfn ../trt_nvfp4 $T/as_v1/P3
  $PY_ANALYSIS -c "import sys, frost.analysis.thor_analysis as t
t.BOOT_SEED = 20261031
sys.argv = ['thor_analysis', '--thor', '$T/as_v1', '--range', '0', '400', '--out', '$A/thor_trt_frost400_analysis_v0v1v2_v1_20261031']
t.main()"
  log "Exploratory: same-noise decision flips between precisions"
  $PY_ANALYSIS -m frost.analysis.thor_precision_flips --out $A/thor_trt_precision_flips_v1_20261031
  log "Addendum 3: compiled VLM variants (comp_fp8, comp_fp8pre_awqfulldec) against V1/V3"
  $PY_ANALYSIS -m frost.analysis.thor_addendum3_analysis --out $A/thor_addendum3_analysis_v1_20261101
fi
