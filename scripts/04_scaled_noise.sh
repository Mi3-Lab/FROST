#!/bin/bash
# Study 4 - scaled-noise control: Euler-10 started from tau*z against the reference curve (Alpamayo-1.5 uses the
# Euler-10 x tau configs already in the pooled probe). Protocol: protocols/frost_main.md (addendum 17).
# Released output: results/scaled_noise_control_a15_v1_20261029. The R1/A2 runs feed the robustness study (13).
source "$(dirname "$0")/common.sh"
H=$D/physicalai_heldout_contexts_v3_20261027
C="euler:10:0.9 euler:10:0.8 euler:10:0.7 euler:10:0.6 euler:10:0.5 euler:10:0.4"

if stage probe; then
  for m in r1 a2; do
    echo "PROBE_ARGS=\"--configs $C\" sbatch --gres=gpu:<GPU>:2 scripts/slurm/probe_multi.sbatch $m $A/scaled_e10_${m}_heldout_v3_20261029 0 400"
  done
fi

if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.scaled_noise_control --data $H --probe $A/temperature_probe_a15_heldout_pooled400_v3_20261028 \
    --out $A/scaled_noise_control_a15_v1_20261029
  # Pooled probe + Euler-10 tau<1 shards -> one probe directory (keys are disjoint), used by frost.analysis.robustness.
  for m in r1 a2; do
    [ -f $A/scaled_e10_${m}_heldout_v3_20261029_c3/manifest.json ] && \
    $PY_ANALYSIS scripts/tools/combine_probes.py --out $A/scaled_noise_${m}_combined_v1_20261031 \
      $A/temperature_probe_${m}_heldout_pooled400_v3_20261028 $A/scaled_e10_${m}_heldout_v3_20261029_c{0,1,2,3}
  done
fi
