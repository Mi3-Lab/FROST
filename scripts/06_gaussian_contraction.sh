#!/bin/bash
# Study 6 - Gaussian contraction prediction: the contraction of k-step Euler predicted for a Gaussian target, compared
# with the native-solver plans of Alpamayo-1.5 on the original 200-clip test sample.
# Released output: results/gaussian_contraction_v1_20261026. The native-solver probe also feeds study 16.
source "$(dirname "$0")/common.sh"
if stage probe; then
  $PY_ALPAMAYO -m frost.probes.native_solver_probe --data $D/physicalai_test_contexts_v1_20261022 \
    --out $A/native_solver_physicalai_v1_20261022 --action-seeds 9201 9202 \
    --solvers euler:4 euler:8 euler:10 heun:5 rk4:2 early_euler:10 cached_euler:8 heun:160
fi
if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.gaussian_contraction_prediction --predictions $A/native_solver_physicalai_v1_20261022 \
    --out $A/gaussian_contraction_v1_20261026
fi
