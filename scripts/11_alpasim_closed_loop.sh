#!/bin/bash
# Study 11 - closed loop in AlpaSim: Alpamayo-1.5 with five sampler configurations on 60 public scenes whose clips are in
# the PhysicalAI-AV test split (58 common scenes after two invalid routes). Protocol: protocols/closed_loop_alpasim.md.
# Setup: clone NVlabs/alpasim at the commit in THIRD_PARTY.md into $ALPASIM_ROOT and apply the sampler hook:
#   git -C $ALPASIM_ROOT apply $PWD/patches/alpasim/frost_sampler.patch
# The hook is inactive unless FROST_METHOD is set, so the official configuration (Euler-10) is unchanged.
# Machine: one GPU shared by the AlpaSim services (original: RTX PRO 6000 Blackwell 96 GB) + Docker.
# Released outputs: results/closedloop_analysis_v1_20261028, results/closedloop_comfort_v1_20261028,
# results/closedloop_exploratory.json (reproduce/tables/make_closedloop_table.py --recompute).
source "$(dirname "$0")/common.sh"
mkdir -p $A/alpasim
if stage data; then
  $PY_ANALYSIS scripts/tools/make_closedloop_scenes.py --suite public_2604 --seed 20261026 --n 60 \
    --alpasim $ALPASIM_ROOT --out $A/alpasim/closedloop_scenes_v1.json
fi
if stage probe; then
  # Resumable: re-run until all five configurations have aggregate (or >= 90% per-rollout) metrics.
  $PY_ANALYSIS -m frost.probes.alpasim_closedloop_grid --scenes $A/alpasim/closedloop_scenes_v1.json --out $A/alpasim/closedloop_v1 \
    --alpasim $ALPASIM_ROOT --extra driver.model.image_decode_device=cpu
fi
if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.closedloop_analysis --grid $A/alpasim/closedloop_v1 --out $A/closedloop_analysis_v1_20261028
  $PY_ANALYSIS -m frost.analysis.closedloop_comfort --grid $A/alpasim/closedloop_v1 --out $A/closedloop_comfort_v1_20261028
  FROST_ARTIFACTS=$A $PY_ANALYSIS reproduce/tables/make_closedloop_table.py --recompute
fi
