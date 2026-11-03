#!/bin/bash
# Study 9 - Qwen-Drive-1.0-4B: v2 probe on the original 60 test clips (RL and SFT planners) and the registered v3 probe
# on the 400 held-out clips (RL planner, native 5 s horizon, SimWAM command rule), with the Heun-3/5 control (addendum 1)
# and Euler-6 (addendum 2). Protocol: protocols/qwen_drive.md. Probes need $QWENDRIVE_SRC/src on PYTHONPATH.
# Released outputs: results/qwendrive_analysis_v2_20261027, results/qwendrive_v3_{analysis,heun_analysis,e6_analysis,curve_band}_20261102.
source "$(dirname "$0")/common.sh"
T=$D/physicalai_test_contexts_v1_20261022/records.json
H=$D/physicalai_heldout_contexts_v3_20261027/records.json
export PYTHONPATH=$PYTHONPATH:$QWENDRIVE_SRC/src

if stage probe; then
  for planner in rl sft; do
    $PY_QWENDRIVE -m frost.probes.qwendrive_temperature_probe_v2 --records $T --out $A/qwendrive_probe_${planner}_v2_20261027 \
      --model $FROST_QWENDRIVE_MODEL --planner planner-$planner --range 100 160 --expert-fp32 --resume
  done
  echo "sbatch scripts/slurm/qwen_v3.sbatch planner-rl $A/qwendrive_v3_rl_heldout_20261101 simwam"
  echo "sbatch scripts/slurm/qwen_v3_heun.sbatch planner-rl $A/qwendrive_v3_heun_rl_heldout_20261102"
  echo "sbatch scripts/slurm/qwen_v3_heun.sbatch planner-rl $A/qwendrive_v3_e6_rl_heldout_20261102 euler:6:1,euler:10:1"
fi

if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.qwendrive_analysis --probe $A/qwendrive_probe_rl_v2_20261027 $A/qwendrive_probe_sft_v2_20261027 \
    --records $T --out $A/qwendrive_analysis_v2_20261027
  V3="$A/qwendrive_v3_rl_heldout_20261101_c0 $A/qwendrive_v3_rl_heldout_20261101_c1 $A/qwendrive_v3_rl_heldout_20261101_c2 $A/qwendrive_v3_rl_heldout_20261101_c3"
  $PY_ANALYSIS -m frost.analysis.qwendrive_v3_analysis --probe $V3 --records $H --out $A/qwendrive_v3_analysis_20261102
  $PY_ANALYSIS -m frost.analysis.qwendrive_v3_heun_analysis --probe $V3 --heun $A/qwendrive_v3_heun_rl_heldout_20261102_c{0,1,2,3} \
    --records $H --out $A/qwendrive_v3_heun_analysis_20261102
  $PY_ANALYSIS -m frost.analysis.qwendrive_v3_e6_analysis --probe $V3 --e6 $A/qwendrive_v3_e6_rl_heldout_20261102_c{0,1,2,3} \
    --records $H --out $A/qwendrive_v3_e6_analysis_20261102
  $PY_ANALYSIS -m frost.analysis.qwendrive_v3_curve_band --probe $V3 --records $H --out $A/qwendrive_v3_curve_band_20261102
fi
