#!/bin/bash
# Qwen-Drive reference-convergence diagnostic (Heun 32..512, RK45 1e-5/1e-7), tau 1 and 0.6, one GPU.
# Usage: OUT=<dir> sbatch scripts/slurm/qwen_conv_ext.sbatch --range A B
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.log
source scripts/common.sh
export PYTHONPATH=$PYTHONPATH:$QWENDRIVE_SRC/src
$PY_QWENDRIVE -m frost.probes.qwendrive_convergence_ext --records $FROST_DATA/physicalai_heldout_contexts_v3_20261027/records.json --model $FROST_QWENDRIVE_MODEL --out ${OUT:?set OUT} "$@"
