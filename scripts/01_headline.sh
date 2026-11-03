#!/bin/bash
# Study 1 - headline matched-temperature test: Alpamayo-1.5, Alpamayo-R1 and Alpamayo-2 on the 400 held-out clips
# (primary), plus the original 60 test-split clips (appendix). Protocol: protocols/frost_main.md.
# Probes: 1 GPU with >= 80 GB for Alpamayo-2 (original: RTX PRO 6000 Blackwell 96 GB workstation; Alpamayo-2 shards on
# 2x H200 NVL / 2x Blackwell via scripts/slurm/probe_multi.sbatch). Analyses: CPU, minutes.
# Released outputs: results/paper_numbers_* (summary.json) and results/paper_numbers_heldout_pooled400_v2_20261028.macros.txt
# (the --macros file of the pooled run, released with a .txt extension)
source "$(dirname "$0")/common.sh"
H=$D/physicalai_heldout_contexts_v3_20261027
T=$D/physicalai_test_contexts_v1_20261022

if stage probe; then
  log "Original 60 test-split clips (context_100-159)"
  $PY_ALPAMAYO -m frost.probes.temperature_probe --data $T --out $A/temperature_probe_v1_20261022
  $PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model r1 --data $T --out $A/temperature_probe_r1_v1_20261022 --range 100 160 --resume
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model a2 --data $T --out $A/temperature_probe_a2_v1_20261022 --range 100 160 --resume

  log "Held-out clips 0-119 (first registered sample)"
  $PY_ALPAMAYO -m frost.probes.temperature_probe --data $H --out $A/temperature_probe_heldout_v3_20261027 --range 0 120 --resume
  M=$A/temperature_probe_r1_heldout_v3_20261027
  shards "$PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model r1 --data $H --out {out} --range {a} {b} --resume" \
    "0 83 $M" "83 102 ${M}_shard2" "102 120 ${M}_shard3"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_shard2 ${M}_shard3 --range 0 120
  # Alpamayo-2 (cluster): resubmit until every shard _c0.._c3 has manifest.json, then merge.
  M=$A/temperature_probe_a2_heldout_v3_20261027
  echo "sbatch --gres=gpu:<GPU>:2 scripts/slurm/probe_multi.sbatch a2 $M 0 120"
  [ -f ${M}_c3/manifest.json ] && { [ -d $M ] || cp -r ${M}_c0 $M; $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_c1 ${M}_c2 ${M}_c3 --range 0 120; }

  log "Held-out clips 120-399 (extension)"
  M=$A/temperature_probe_a15_heldout_ext_v3_20261028
  shards "$PY_ALPAMAYO -m frost.probes.temperature_probe --data $H --out {out} --range {a} {b} --resume" \
    "120 213 $M" "213 306 ${M}_shard2" "306 400 ${M}_shard3"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_shard2 ${M}_shard3 --range 120 400
  M=$A/temperature_probe_r1_heldout_ext_v3_20261027
  shards "$PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model r1 --data $H --out {out} --range {a} {b} --resume" \
    "120 213 $M" "213 306 ${M}_shard2" "306 400 ${M}_shard3"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_shard2 ${M}_shard3 --range 120 400
  M=$A/temperature_probe_a2_heldout_ext_v3_20261027
  echo "sbatch --gres=gpu:<GPU>:2 scripts/slurm/probe_multi.sbatch a2 $M 120 400   # original: H200 NVL first, Blackwell for clips >= 141"
  [ -f ${M}_c3/manifest.json ] && { [ -d $M ] || cp -r ${M}_c0 $M; $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_c1 ${M}_c2 ${M}_c3 --range 120 400; }
fi

if stage merge; then
  log "Pooled 400 clips = first sample + extension (file merge only)"
  P=$A/temperature_probe_a15_heldout_pooled400_v3_20261028; [ -d $P ] || cp -r $A/temperature_probe_heldout_v3_20261027 $P
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $P --shards $A/temperature_probe_a15_heldout_ext_v3_20261028 --range 0 400
  P=$A/temperature_probe_r1_heldout_pooled400_v3_20261028; [ -d $P ] || cp -r $A/temperature_probe_r1_heldout_v3_20261027 $P
  # The extension ran with a later probe version that only adds --configs; its source hash is accepted explicitly.
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $P --shards $A/temperature_probe_r1_heldout_ext_v3_20261027 --range 0 400 \
    --allow-source $($PY_ANALYSIS -c "import json;print(json.load(open('$A/temperature_probe_r1_heldout_ext_v3_20261027/manifest.json'))['source_sha256'])")
  P=$A/temperature_probe_a2_heldout_pooled400_v3_20261028; [ -d $P ] || cp -r $A/temperature_probe_a2_heldout_v3_20261027 $P
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $P --shards $A/temperature_probe_a2_heldout_ext_v3_20261027 --range 0 400
fi

if stage analysis; then
  log "Matched-temperature statistics (2000 clip resamples, seed 20261026)"
  pn() { $PY_ANALYSIS -m frost.analysis.paper_numbers --artifacts $A "$@"; }
  pn --data $T --out $A/paper_numbers_v4_20261026 --macros $A/paper_numbers_v4_20261026_numbers.tex --boot 1000
  pn --data $H --out $A/paper_numbers_heldout_a15_v1_20261027 --macros $A/paper_numbers_heldout_a15_v1_20261027.macros.tex --range 0 120 --boot 2000 --probes alpamayo15=temperature_probe_heldout_v3_20261027
  pn --data $H --out $A/paper_numbers_heldout_r1_v1_20261027 --macros $A/paper_numbers_heldout_r1_v1_20261027.macros.tex --range 0 120 --boot 2000 --probes alpamayo_r1=temperature_probe_r1_heldout_v3_20261027
  pn --data $H --out $A/paper_numbers_heldout_a2_v1_20261027 --macros $A/paper_numbers_heldout_a2_v1_20261027.macros.tex --range 0 120 --probes alpamayo2=temperature_probe_a2_heldout_v3_20261027
  pn --data $H --out $A/paper_numbers_heldout_ext_v1_20261028 --macros $A/paper_numbers_heldout_ext_v1_20261028.macros.tex --range 120 400 --probes alpamayo_r1=temperature_probe_r1_heldout_ext_v3_20261027 alpamayo2=temperature_probe_a2_heldout_ext_v3_20261027
  pn --data $H --out $A/paper_numbers_heldout_ext_a2blackwell_v1_20261028 --macros $A/paper_numbers_heldout_ext_a2blackwell_v1_20261028.macros.tex --range 141 400 --probes alpamayo2=temperature_probe_a2_heldout_ext_v3_20261027
  pn --data $H --out $A/paper_numbers_heldout_ext_v2_20261028 --macros $A/paper_numbers_heldout_ext_v2_20261028.macros.tex --range 120 400 --probes alpamayo15=temperature_probe_a15_heldout_ext_v3_20261028 alpamayo_r1=temperature_probe_r1_heldout_ext_v3_20261027 alpamayo2=temperature_probe_a2_heldout_ext_v3_20261027
  pn --data $H --out $A/paper_numbers_heldout_pooled400_v2_20261028 --macros $A/paper_numbers_heldout_pooled400_v2_20261028.macros.tex --range 0 400 --probes alpamayo15=temperature_probe_a15_heldout_pooled400_v3_20261028 alpamayo_r1=temperature_probe_r1_heldout_pooled400_v3_20261028 alpamayo2=temperature_probe_a2_heldout_pooled400_v3_20261028
fi
