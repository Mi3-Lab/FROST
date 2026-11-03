#!/bin/bash
# Qwen-Drive v3 controls (protocols/qwen_drive.md, addenda 1 and 2): 400 held-out clips, 2 GPUs (original: 2x A100),
# 4 shards of 100 clips (task k: shards k, k+2), resumable. Same scenes, command, greedy reasoning, fp32 expert, seeds
# and noise as qwen_v3.sbatch.
# Usage: sbatch scripts/slurm/qwen_v3_heun.sbatch PLANNER OUT_PREFIX [CONFIGS]
#   CONFIGS (addendum 2): method:steps:tau,... passed to --configs; default = the addendum 1 list (Heun-3/5, E10, Heun-32).
#   SHARDS (optional env, e.g. SHARDS="0"): shards to run, for single-GPU jobs.
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:<GPU>:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.log
source scripts/common.sh
export PYTHONPATH=$PYTHONPATH:$QWENDRIVE_SRC/src
export PLANNER=$1 PFX=$2 CFG=${3:+--configs $3}
srun --ntasks=$SLURM_NTASKS --gpus-per-task=1 bash -c '
  for k in ${SHARDS:-$(seq $SLURM_PROCID $SLURM_NTASKS 3)}; do
    a=$(( k*100 )); b=$(( a+100 ))
    $PY_QWENDRIVE -m frost.probes.qwendrive_heun_control_v3 --records $FROST_DATA/physicalai_heldout_contexts_v3_20261027/records.json \
      --model $FROST_QWENDRIVE_MODEL --planner $PLANNER --range $a $b --out ${PFX}_c$k --command simwam --resume $CFG
  done'
