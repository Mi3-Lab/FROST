#!/bin/bash
# Study 15 - overlap of every evaluated clip sample with the Alpamayo-aligned train/test split of PhysicalAI-AV.
# Released outputs: results/split_overlap_v1_20261027, results/split_overlap_confirmation_v1_20261029.
source "$(dirname "$0")/common.sh"
S=$D/physicalai_splits_20261027/splits/alpamayo_aligned.parquet
$PY_ANALYSIS -m frost.data.split_overlap --split $S --out $A/split_overlap_v1_20261027 \
  --sample main $D/physicalai_test_contexts_v1_20261022/records.json 100 160 \
  --sample heldout_primary $D/physicalai_heldout_contexts_v3_20261027/records.json 0 120
$PY_ANALYSIS -m frost.data.split_overlap --split $S --out $A/split_overlap_confirmation_v1_20261029 \
  --sample confirmation $D/physicalai_test_contexts_v2_20261022/records.json 0 436
