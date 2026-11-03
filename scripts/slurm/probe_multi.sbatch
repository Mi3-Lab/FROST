#!/bin/bash
# Alpamayo-R1 / Alpamayo-2 temperature probe on one node with 2 GPUs. The clip range is always split into 4 contiguous
# shards OUT_PREFIX_c{0..3}; task k runs shards k, k+2 in turn (resumable), so the outputs do not depend on the GPU count.
# Usage: sbatch scripts/slurm/probe_multi.sbatch MODEL OUT_PREFIX START STOP [DATA]   (extra probe options via PROBE_ARGS)
# The original runs used one-hour jobs on 2x H200 NVL or 2x Blackwell GPUs and were resubmitted until every shard had
# its manifest.json.
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:<GPU>:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=00:59:00
#SBATCH --output=slurm-%x-%j.log
source scripts/common.sh
export MODEL=$1 PREFIX=$2 START=$3 STOP=$4 DATA=${5:-$FROST_DATA/physicalai_heldout_contexts_v3_20261027}
srun --ntasks=$SLURM_NTASKS --gpus-per-task=1 bash -c '
  n=$(( (STOP-START+3)/4 ))
  for k in $(seq $SLURM_PROCID $SLURM_NTASKS 3); do
    a=$(( START+k*n )); b=$(( a+n )); [ $b -gt $STOP ] && b=$STOP
    echo "task $SLURM_PROCID gpu $CUDA_VISIBLE_DEVICES shard $k clips $a-$b"
    $PY_ALPAMAYO -m frost.probes.multimodel_temperature_probe --model $MODEL --data $DATA --out ${PREFIX}_c$k --range $a $b --resume $PROBE_ARGS
  done'
