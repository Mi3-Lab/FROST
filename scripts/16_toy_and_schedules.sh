#!/bin/bash
# Study 16 - toy model of mode collapse under few-step Euler, and the equal-NFE schedule study on Alpamayo-1.5
# (calibration on clips 0-99 of the test sample, evaluation on 100-199, confirmation on the disjoint 436-clip sample;
# matched-temperature comparison of the schedules). Released outputs: results/mode_collapse_theory_v1_20261022,
# results/schedule_eval_physicalai_v2_20261022 (reduced rows.json), results/schedule_matched_report_v1_20261022.
source "$(dirname "$0")/common.sh"
T=$D/physicalai_test_contexts_v1_20261022
P=$A/native_solver_physicalai_v1_20261022   # scripts/06_gaussian_contraction.sh
CURVES=$A/schedule_curves_physicalai_v1_20261022.json

if stage analysis-toy || stage analysis; then
  $PY_ANALYSIS -m frost.analysis.mode_collapse_theory --out $A/mode_collapse_theory_v1_20261022
fi
if stage probe; then
  $PY_ALPAMAYO -m frost.probes.commitment_probe --data $T --out $A/commitment_physicalai_v2_20261022 --no-kick
  $PY_ALPAMAYO -m frost.probes.schedule_eval --phase calibrate --data $T --commitment $A/commitment_physicalai_v2_20261022 \
    --solver-predictions $P --curves $CURVES
  $PY_ALPAMAYO -m frost.probes.schedule_eval --phase evaluate --data $D/physicalai_test_contexts_v2_20261022 --solver-predictions $P \
    --curves $CURVES --out $A/schedule_eval_physicalai_v2_20261022 --evaluation 0 436 --seeds 9201 9202 9203 9204 --inline-reference 80
  # Matched-temperature comparison on the 100 evaluation clips, run in three restarts (100-199, 123-199, 130-199) and merged.
  for spec in "v1 100" "v1b 123" "v1c 130"; do set -- $spec
    $PY_ALPAMAYO -m frost.probes.schedule_eval --phase evaluate --data $T --solver-predictions $P --curves $CURVES \
      --out $A/schedule_eval_matched_$1_20261022 --evaluation $2 200 --seeds 9201 9202 9203 9204 9205 9206 --inline-reference 32 --save-plans
  done
  $PY_ANALYSIS scripts/tools/merge_plans.py --out $A/schedule_eval_matched_merged_v1_20261022 \
    $A/schedule_eval_matched_v1_20261022 $A/schedule_eval_matched_v1b_20261022 $A/schedule_eval_matched_v1c_20261022
fi
if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.schedule_matched_report --data $T --temperature $A/temperature_probe_v1_20261022 \
    --schedules $A/schedule_eval_matched_merged_v1_20261022 --out $A/schedule_matched_report_v1_20261022
fi
