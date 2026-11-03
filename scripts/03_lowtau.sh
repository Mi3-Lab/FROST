#!/bin/bash
# Study 3 - low-temperature extension of the reference curve (tau = 0.3, 0.2, 0.1) for one- and two-step Euler,
# best-reference and mean-of-reference baselines. Protocol: protocols/frost_main.md (addendum 16).
# Released outputs: results/lowtau_{analysis,best_reference,mean_baseline}_{a15,r1}_v1_20261029
source "$(dirname "$0")/common.sh"
H=$D/physicalai_heldout_contexts_v3_20261027
C="heun:32:0.3 heun:32:0.2 heun:32:0.1"

if stage probe; then
  M=$A/lowtau_ref_a15_heldout_v3_20261028
  shards "$PY_ALPAMAYO -m frost.probes.temperature_probe --data $H --out {out} --range {a} {b} --resume --configs $C" \
    "0 100 $M" "100 200 ${M}_shard2" "200 300 ${M}_shard3" "300 400 ${M}_shard4"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_shard2 ${M}_shard3 ${M}_shard4 --range 0 400
  echo "PROBE_ARGS=\"--configs $C\" sbatch --gres=gpu:<GPU>:2 scripts/slurm/probe_multi.sbatch r1 $A/lowtau_ref_r1_heldout_v3_20261028 0 400"
  M=$A/lowtau_ref_r1_cluster_heldout_v3_20261029
  [ -f $A/lowtau_ref_r1_heldout_v3_20261028_c3/manifest.json ] && { [ -d $M ] || cp -r $A/lowtau_ref_r1_heldout_v3_20261028_c0 $M
    $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards $A/lowtau_ref_r1_heldout_v3_20261028_c{1,2,3} --range 0 400; }
fi

if stage analysis; then
  declare -A LOW=([a15]=lowtau_ref_a15_heldout_v3_20261028 [r1]=lowtau_ref_r1_cluster_heldout_v3_20261029)
  for m in a15 r1; do
    for mod in lowtau_analysis lowtau_best_reference lowtau_mean_baseline; do
      $PY_ANALYSIS -m frost.analysis.$mod --data $H --main $A/temperature_probe_${m}_heldout_pooled400_v3_20261028 \
        --lowtau $A/${LOW[$m]} --out $A/${mod}_${m}_v1_20261029
    done
  done
fi
