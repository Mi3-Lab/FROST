#!/bin/bash
# SimWAM reference-convergence extension: Heun-128/256/512 and adaptive RK45 (rtol=atol 1e-5 / 1e-7), one GPU (original: 1x A100).
# Usage: OUT=<dir> sbatch scripts/slurm/simwam_extend.sbatch
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.log
source scripts/common.sh
export HF_HUB_OFFLINE=1
mkdir -p ${OUT:?set OUT}
CLIPS=${CLIPS:-context_000 context_072 context_130 context_010 context_268}
$PY_SIMWAM -m frost.probes.simwam_temperature_probe --extend --device cuda --ego $FROST_DATA/simwam_ego_inputs_20261031/ego.npz --clips $CLIPS --seeds 9201 9202 --max-nfe ${MAXNFE:-2000} --out $OUT
