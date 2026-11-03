#!/bin/bash
# Study 18 - data preparation: PhysicalAI-AV context caches, splits, WOD-E2E contexts, SimWAM ego inputs.
# Machine: one GPU workstation for the PhysicalAI caches (CPU-bound; GPU unused), CPU for the rest.
# Outputs: $FROST_DATA/<name> (names as in results/ and the commands below).
source "$(dirname "$0")/common.sh"

if stage physicalai; then
  log "PhysicalAI-AV test sample (200 clips, one per chunk, seed 20261022) used by the original 60-clip study"
  $PY_ALPAMAYO -m frost.data.physicalai_contexts --out $D/physicalai_test_contexts_v1_20261022 --count 200 --workers 6
  log "Confirmation sample (436 further clips, disjoint chunks, seed 20261023) used by the schedule confirmation"
  $PY_ALPAMAYO -m frost.data.physicalai_contexts --out $D/physicalai_test_contexts_v2_20261022 --count 436 --seed 20261023 \
    --workers 8 --exclude-records $D/physicalai_test_contexts_v1_20261022/records.json
  log "Alpamayo-aligned split files"
  $PY_ANALYSIS -I -c "from huggingface_hub import hf_hub_download
for f in ['splits/alpamayo_aligned.parquet','splits/default.parquet']:
    hf_hub_download('nvidia/PhysicalAI-Autonomous-Vehicles',f,repo_type='dataset',local_dir='$D/physicalai_splits_20261027')"
  log "400 held-out clips (test partition of the Alpamayo-aligned split, seed 20261027): the primary sample"
  $PY_ALPAMAYO -m frost.data.physicalai_contexts --out $D/physicalai_heldout_contexts_v3_20261027 --count 400 --seed 20261027 \
    --aligned-split $D/physicalai_splits_20261027/splits/alpamayo_aligned.parquet
fi

if stage wod; then
  log "WOD-E2E validation: index all frames, keep rated frames (sorted by segment), convert"
  $PY_ALPAMAYO -m frost.data.wod_e2e_contexts --data $WOD_E2E_ROOT scan --shards $(seq 0 92) --out $D/wod_e2e_val_index_all_v1_20261028.jsonl
  $PY_ANALYSIS scripts/tools/wod_rated_index.py $D/wod_e2e_val_index_all_v1_20261028.jsonl $D/wod_e2e_val_index_rated_v1_20261029.jsonl
  $PY_ALPAMAYO -m frost.data.wod_e2e_contexts --data $WOD_E2E_ROOT convert --index $D/wod_e2e_val_index_rated_v1_20261029.jsonl \
    --history-index $D/wod_e2e_val_index_all_v1_20261028.jsonl --out $D/wod_e2e_val_contexts_v2_20261029
fi

if stage simwam; then
  log "SimWAM ego inputs for the 400 held-out clips (Qwen-Drive environment: PhysicalAI-AV reader)"
  PYTHONPATH=$PYTHONPATH:$QWENDRIVE_SRC/src $PY_QWENDRIVE -m frost.data.simwam_ego_inputs \
    --records $D/physicalai_heldout_contexts_v3_20261027/records.json --out $D/simwam_ego_inputs_20261031/ego.npz
fi
