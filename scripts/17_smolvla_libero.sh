#!/bin/bash
# Study 17 - manipulation (secondary domain): SmolVLA on the four LIBERO suites, Euler-10/4/2/1 at tau=1 and Heun-16 at
# tau in {1, 0.9, 0.75, 0.5, 0.25}; 10 episodes per task, eval seed 1000. Protocol: protocols/smolvla_libero.md.
# Environment: LeRobot + LIBERO ($PY_ROBOT), MuJoCo EGL rendering, one GPU shared by up to three workers.
# Released outputs: results/robot_grid_smolvla_v1_20261022 (eval_info.json per config and suite),
# results/robot_smoke_20261022 (contraction statistics of the probe run).
source "$(dirname "$0")/common.sh"
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=$LIBERO_CONFIG
if stage probe; then
  # Smoke run with plan probes every 5 calls (input of the contraction statistics in the appendix).
  FROST_METHOD=euler FROST_STEPS=10 FROST_TAU=1 FROST_PROBE_EVERY=5 FROST_PROBE_OUT=$A/robot_smoke_20261022/probe.npz \
  $PY_ROBOT frost/probes/robot_temperature_eval.py --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_spatial \
    --env.task_ids="[0]" --eval.n_episodes=2 --eval.batch_size=1 --output_dir=$A/robot_smoke_20261022/eval --seed=1000
  # Full grid (resumable; each (config, suite) is skipped once its eval_info.json exists).
  $PY_ROBOT -P -m frost.probes.robot_temperature_grid --policy HuggingFaceVLA/smolvla_libero --out $A/robot_grid_smolvla_v1_20261022 --parallel 3
fi
