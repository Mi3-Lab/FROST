#!/bin/bash
# Study 5 - convergence of the reference sampler: Heun-64/128 against Heun-32 for Alpamayo-1.5/R1/2 (addenda 18-20,
# protocols/frost_main.md), and the extension to Qwen-Drive, SimWAM and Cosmos 3 Nano with Heun-128..512 and adaptive
# RK45 (frost.probes.adaptive_ref). Released outputs: results/convergence_a19_analysis_v1_20261030,
# results/convergence_a20_analysis_{r1_v1_20261030,a2_v1_20261031}, results/convergence_ext_summary_{qwen,simwam}_20261101,
# results/convergence_ext_summary_cosmos_c01234_20261102.
source "$(dirname "$0")/common.sh"
H=$D/physicalai_heldout_contexts_v3_20261027
R=$H/records.json

if stage probe; then
  log "Alpamayo-1.5, clips 0-99: Heun-64/128 on a tau grid plus a finer Heun-32 grid (addendum 18)"
  C="$(for t in 1.0 0.9 0.85 0.8 0.75 0.7; do echo -n "heun:64:$t heun:128:$t "; done)heun:32:0.875 heun:32:0.85 heun:32:0.825 heun:32:0.775 heun:32:0.75"
  M=$A/convergence_a18_a15_heldout_v3_20261029
  shards "$PY_ALPAMAYO -m frost.probes.temperature_probe --data $H --out {out} --range {a} {b} --resume --configs $C" \
    "0 25 $M" "25 50 ${M}_s2" "50 75 ${M}_s3" "75 100 ${M}_s4"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_s2 ${M}_s3 ${M}_s4 --range 0 100

  log "Alpamayo-1.5, clips 100-399: Heun-64 grid (addendum 19)"
  C="$(for t in 1.0 0.9 0.85 0.8 0.75 0.7; do echo -n "heun:64:$t "; done)"
  M=$A/convergence_a19_a15_heun64_heldout_v3_20261030
  shards "$PY_ALPAMAYO -m frost.probes.temperature_probe --data $H --out {out} --range {a} {b} --resume --configs $C" \
    "100 175 $M" "175 250 ${M}_s2" "250 325 ${M}_s3" "325 400 ${M}_s4"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_s2 ${M}_s3 ${M}_s4 --range 100 400

  log "Alpamayo-R1 and -2, 400 clips: Euler-10 and the Heun-64 grid (addendum 20)"
  C="euler:10:1.0 $(for t in 1.0 0.9 0.85 0.8 0.75 0.7; do echo -n "heun:64:$t "; done)"
  M=$A/convergence_a20_r1_heun64_heldout_v3_20261030
  shards "$PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model r1 --data $H --out {out} --range {a} {b} --resume --configs $C" \
    "0 100 $M" "100 200 ${M}_s2" "200 300 ${M}_s3" "300 334 ${M}_s4" "334 367 ${M}_s5" "367 400 ${M}_s6"
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_s2 ${M}_s3 ${M}_s4 ${M}_s5 ${M}_s6 --range 0 400
  M=$A/convergence_a20_a2_heun64_heldout_v3_20261030   # one process at a time: Alpamayo-2 needs about 72 GB
  for spec in "0 134 $M" "134 200 ${M}_s2" "300 400 ${M}_s3" "250 300 ${M}_s4" "200 250 ${M}_s5"; do set -- $spec
    $PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model a2 --data $H --out $3 --range $1 $2 --resume --configs $C; done
  $PY_ANALYSIS -m frost.data.merge_probe_shards --main $M --shards ${M}_s2 ${M}_s3 ${M}_s4 ${M}_s5 --range 0 400

  log "Other policies (cluster jobs; see scripts/slurm)"
  echo "OUT=$A/qwendrive_convergence_ext_smoke sbatch scripts/slurm/qwen_conv_ext.sbatch --range 0 2"
  echo "OUT=$A/qwendrive_convergence_ext_t     sbatch scripts/slurm/qwen_conv_ext.sbatch --range 2 7"
  echo "OUT=$A/qwendrive_convergence_ext_t2    sbatch scripts/slurm/qwen_conv_ext.sbatch --range 5 7"
  echo "OUT=$A/qwendrive_convergence_ext_g     sbatch scripts/slurm/qwen_conv_ext.sbatch --range 7 20"
  echo "OUT=$A/simwam_extend_20261101_all sbatch scripts/slurm/simwam_extend.sbatch"
  echo "sbatch scripts/slurm/cosmos3nano.sbatch conv $A/cosmos3nano_conv_20261031 --clips 0 1"
  echo "sbatch scripts/slurm/cosmos3nano.sbatch conv $A/cosmos3nano_conv_20261031 --clips 2 3 4"
  echo "sbatch scripts/slurm/cosmos3nano.sbatch ext  $A/cosmos3nano_ext_20261101    --clips 0 1 2"
  echo "sbatch scripts/slurm/cosmos3nano.sbatch ext  $A/cosmos3nano_ext_20261101_c3 --clips 3"
  echo "sbatch scripts/slurm/cosmos3nano.sbatch ext  $A/cosmos3nano_ext_20261101_c4 --clips 4"
fi

if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.convergence_a19_analysis --out $A/convergence_a19_analysis_v1_20261030
  $PY_ANALYSIS -m frost.analysis.convergence_a20_analysis --model r1 --out $A/convergence_a20_analysis_r1_v1_20261030
  $PY_ANALYSIS -m frost.analysis.convergence_a20_analysis --model a2 --out $A/convergence_a20_analysis_a2_v1_20261031
  $PY_ANALYSIS -m frost.analysis.convergence_ext_analysis qwen --ext $A/qwendrive_convergence_ext_{smoke,t,t2,g} \
    --out $A/convergence_ext_summary_qwen_20261101
  $PY_ANALYSIS -m frost.analysis.convergence_ext_analysis simwam --ext $A/simwam_extend_20261101_all/extend.json \
    --out $A/convergence_ext_summary_simwam_20261101
  $PY_ANALYSIS -m frost.analysis.convergence_ext_analysis cosmos --ext $A/cosmos3nano_ext_20261101 $A/cosmos3nano_ext_20261101_c3 \
    $A/cosmos3nano_ext_20261101_c4 --conv $A/cosmos3nano_conv_20261031 --records $R --out $A/convergence_ext_summary_cosmos_c01234_20261102
fi
