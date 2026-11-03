#!/bin/bash
# Study 8 - second planner family on NAVSIM navtest (500 scenes): ReCogDrive IL/RL (DDIM) and DiffusionDrive v1, scored
# with the PDM score. Protocol: protocols/navsim.md (addenda 1, 1b, 2). Environments: ReCogDrive Python 3.10
# ($PY_RECOGDRIVE), DiffusionDrive/NAVSIM Python 3.9 ($PY_DIFFUSIONDRIVE, also used for scoring and analysis).
# Released outputs: results/navsim_analysis_recogdrive_{rl,il}_addendum2_v1_20261028, results/navsim_analysis_recogdrive_rl_v{1,2}_20261028,
# results/navsim_analysis_diffusiondrive_v3_20261029.
source "$(dirname "$0")/common.sh"
source configs/navsim.env
DD=$DIFFUSIONDRIVE_SRC

if stage data; then
  log "Scene sample (seed 20261028, 500 navtest scenes) and the NAVSIM metric cache"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYTHONPATH:$DD $PY_DIFFUSIONDRIVE -m frost.data.navsim_scene_sample --repo $DD --out $A/navsim_scenes_v1_20261028.json
  (cd $DD && CUDA_VISIBLE_DEVICES= PYTHONPATH=$DD $PY_DIFFUSIONDRIVE navsim/planning/script/run_metric_caching.py \
     train_test_split=navtest worker=ray_distributed worker.threads_per_node=22 cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache)
fi

probe() {  # probe <model> <out> [extra args]; then PDM-score the stored plans
  local model=$1 out=$2; shift 2
  local src=$RECOGDRIVE_SRC py=$PY_RECOGDRIVE
  [ $model = diffusiondrive ] && { src=$DD; py=$PY_DIFFUSIONDRIVE; }
  PYTHONPATH=$PYTHONPATH:$src $py -m frost.probes.navsim_temperature_probe --model $model --scenes $A/navsim_scenes_v1_20261028.json --out $out --resume "$@"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYTHONPATH:$DD $PY_DIFFUSIONDRIVE -m frost.probes.navsim_pdm_score_plans --repo $DD \
    --scenes $A/navsim_scenes_v1_20261028.json --plans $out --workers 16 --resume
}

if stage probe; then
  probe recogdrive_rl $A/navsim_probe_recogdrive_rl_v1_20261028
  probe recogdrive_il $A/navsim_probe_recogdrive_il_v1_20261028
  probe diffusiondrive $A/navsim_probe_diffusiondrive_v1_20261028
  # Addendum 2: leading-grid DDIM sweep against the 100-step reference, with a check that the release sampler is reproduced.
  probe recogdrive_rl $A/navsim_probe_recogdrive_rl_addendum2_v1_20261028 --config-set addendum2 --verify-release
  probe recogdrive_il $A/navsim_probe_recogdrive_il_addendum2_v1_20261028 --config-set addendum2 --verify-release
fi

if stage analysis; then
  an() { PYTHONPATH=$PYTHONPATH:$DD $PY_DIFFUSIONDRIVE -m frost.analysis.navsim_analysis --cache $NAVSIM_EXP_ROOT/metric_cache "$@"; }
  # rl_v1 and rl_v2 were written by earlier versions of navsim_analysis (docs/PROVENANCE.md): v1 has the registered raw
  # 8-point flips only; v2 added the dense-plan flips. The current module writes both (flip_registered and flip).
  an --probe $A/navsim_probe_recogdrive_rl_v1_20261028 --family recogdrive --out $A/navsim_analysis_recogdrive_rl_v1_20261028
  an --probe $A/navsim_probe_recogdrive_rl_v1_20261028 --family recogdrive --out $A/navsim_analysis_recogdrive_rl_v2_20261028
  an --probe $A/navsim_probe_recogdrive_rl_addendum2_v1_20261028 --family recogdrive_lead --out $A/navsim_analysis_recogdrive_rl_addendum2_v1_20261028
  an --probe $A/navsim_probe_recogdrive_il_addendum2_v1_20261028 --family recogdrive_lead --out $A/navsim_analysis_recogdrive_il_addendum2_v1_20261028
  an --probe $A/navsim_probe_diffusiondrive_v1_20261028 --family diffusiondrive --out $A/navsim_analysis_diffusiondrive_v3_20261029
fi
