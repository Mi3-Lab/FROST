#!/bin/bash
# Qwen-Drive v3 probe on the 400 held-out clips: 2 GPUs (original: 2x A100), 4 contiguous shards of 100 clips
# (task k: shards k, k+2), resumable.
# Usage: sbatch scripts/slurm/qwen_v3.sbatch PLANNER OUT_PREFIX [command rule: simwam|legacy]
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:<GPU>:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.log
source scripts/common.sh
export PYTHONPATH=$PYTHONPATH:$QWENDRIVE_SRC/src
export PLANNER=$1 PFX=$2 RULE=${3:-simwam}
srun --ntasks=2 --gpus-per-task=1 bash -c '
  for k in $(seq $SLURM_PROCID 2 3); do
    a=$(( k*100 )); b=$(( a+100 ))
    $PY_QWENDRIVE -m frost.probes.qwendrive_temperature_probe_v3 --records $FROST_DATA/physicalai_heldout_contexts_v3_20261027/records.json \
      --model $FROST_QWENDRIVE_MODEL --planner $PLANNER --range $a $b --out ${PFX}_c$k --command $RULE --resume
  done'
