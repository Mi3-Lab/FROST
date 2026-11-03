#!/bin/bash
# Thor protocol addendum 2: TRT FROST probe on held-out context_000-399. Prefix once per clip (PyTorch bf16),
# then the three batch-6 engines (V0 bf16, V1 FP8, V2 NVFP4 no_ends).
set -u
T=$FROST_THOR_ROOT/trt; OUT=$T/out; R=$T/frost400; source $FROST_THOR_ROOT/env.sh; export PYTHONPATH=$PYTHONPATH:$T/scripts
cd $T/scripts; mkdir -p $R; L=$OUT/run_trt_frost400.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
log start
$T/venv/bin/python thor_trt_probe.py prefix --range 0 400 --resume --out $R > $OUT/frost400_prefix.log 2>&1; log "prefix exit $?"
for ve in trt_bf16:probe_bf16 trt_fp8:probe_fp8 trt_nvfp4:probe_nvfp4; do v=${ve%%:*}; e=$T/engines/${ve#*:}_b6_L3456.engine
  $T/venv/bin/python thor_trt_probe.py field --variant $v --engine $e --range 0 400 --resume --out $R > $OUT/frost400_$v.log 2>&1; log "field $v exit $?"
done
log done
