#!/bin/bash
# Study 2 - Heun-3/Heun-5 controls for Alpamayo-R1 and Alpamayo-2 (same run also regenerates Heun-32 at tau=1, so flips
# are measured against a same-run reference). Protocol: protocols/frost_main.md (addendum 15).
# Released outputs: results/heun_controls_analysis_{r1,a2}_v1_20261028, ..._a2_identicalref_200_400_v1_20261028
source "$(dirname "$0")/common.sh"
H=$D/physicalai_heldout_contexts_v3_20261027
C="heun:3:1.0 heun:5:1.0 heun:32:1.0"

if stage probe; then
  M=$A/heun_controls_r1_heldout_v3_20261028
  shards "$PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model r1 --data $H --out {out} --range {a} {b} --resume --configs $C" \
    "0 133 $M" "133 266 ${M}_shard2" "266 400 ${M}_shard3"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_shard2 ${M}_shard3 --range 0 400
  M=$A/heun_controls_a2_heldout_v3_20261028
  echo "PROBE_ARGS=\"--configs $C\" sbatch --gres=gpu:<GPU>:2 scripts/slurm/probe_multi.sbatch a2 $M 0 400"
  [ -f ${M}_c3/manifest.json ] && { [ -e $M ] || cp -r ${M}_c0 $M; $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_c1 ${M}_c2 ${M}_c3 --range 0 400; }
fi

if stage analysis; then
  for m in r1 a2; do
    $PY_ANALYSIS -m frost.analysis.heun_controls_analysis --data $H --main $A/temperature_probe_${m}_heldout_pooled400_v3_20261028 \
      --controls $A/heun_controls_${m}_heldout_v3_20261028 --out $A/heun_controls_analysis_${m}_v1_20261028
  done
  # Clips 200-399 of Alpamayo-2, where the control run's Heun-32 reference is bit-identical to the main probe's.
  $PY_ANALYSIS -m frost.analysis.heun_controls_analysis --data $H --main $A/temperature_probe_a2_heldout_pooled400_v3_20261028 \
    --controls $A/heun_controls_a2_heldout_v3_20261028 --range 200 400 --out $A/heun_controls_analysis_a2_identicalref_200_400_v1_20261028
fi
