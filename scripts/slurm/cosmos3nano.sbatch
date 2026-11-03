#!/bin/bash
# Cosmos 3 Nano official AV-policy probe (smoke | conv | ext | pilot) on one GPU (original runs: 1x H200 NVL or 1x A100).
# Usage: sbatch [--gres=gpu:<GPU>:1 --time=..] scripts/slurm/cosmos3nano.sbatch MODE OUT [probe options...]
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:<GPU>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.log
source scripts/common.sh
MODE=$1 OUT=$2; shift 2
$PY_COSMOS -m frost.probes.cosmos3nano_probe $MODE --model $FROST_COSMOS3_NANO_MODEL --data $FROST_DATA/physicalai_heldout_contexts_v3_20261027 --out $OUT "$@"
