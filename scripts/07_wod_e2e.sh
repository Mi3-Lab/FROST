#!/bin/bash
# Study 7 - WOD-E2E long tail: Alpamayo-1.5 on the 479 rated validation frames; registered primary sample = first 280
# segments, secondary = all 479; rater feedback score by our re-implementation and by the official Waymo module.
# Protocol: protocols/wod_e2e.md. Released outputs: results/wod_e2e_analysis_{primary,all479}_converged10k_v2_20261029,
# results/wod_e2e_analysis_official_rfs_v1_20261030.
source "$(dirname "$0")/common.sh"
W=$D/wod_e2e_val_contexts_v2_20261029
M=$A/wod_e2e_probe_a15_v1_20261029

if stage probe; then
  C="euler:10:1.0 euler:4:1.0 euler:1:1.0 $(for t in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do echo -n "heun:32:$t "; done)"
  R=(0 70 140 210 280 330 380 430 479)
  # Eight shards, four processes at a time on one GPU.
  shards "$PY_ALPAMAYO -m frost.probes.temperature_probe --data $W --out {out} --range {a} {b} --resume --configs $C" \
    "${R[0]} ${R[1]} ${M}_p0" "${R[1]} ${R[2]} ${M}_p1" "${R[2]} ${R[3]} ${M}_p2" "${R[3]} ${R[4]} ${M}_p3"
  shards "$PY_ALPAMAYO -m frost.probes.temperature_probe --data $W --out {out} --range {a} {b} --resume --configs $C" \
    "${R[4]} ${R[5]} ${M}_p4" "${R[5]} ${R[6]} ${M}_p5" "${R[6]} ${R[7]} ${M}_p6" "${R[7]} ${R[8]} ${M}_p7"
fi

if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.wod_e2e_analysis --data $W --probe "${M}_p{}" --boot 10000 \
    --out $A/wod_e2e_analysis_primary_converged10k_v2_20261029
  $PY_ANALYSIS -m frost.analysis.wod_e2e_analysis --data $W --probe "${M}_p{}" --n 479 --shards 8 --boot 10000 \
    --out $A/wod_e2e_analysis_all479_converged10k_v2_20261029
  # Official score: download rater_feedback_utils.py unmodified (THIRD_PARTY.md) and set WAYMO_RFS_OFFICIAL.
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $PY_ANALYSIS -m frost.analysis.wod_e2e_rfs_official --data $W --probe "${M}_p{}" \
    --shards 8 --boot 10000 --samples 280,479 --out $A/wod_e2e_analysis_official_rfs_v1_20261030
fi
