#!/bin/bash
# Study 13 - exploratory / post hoc robustness of the FROST verdicts (not pre-registered): alternative matching statistics,
# effect fractions, noise subsets, decision-threshold perturbations, Heun diagnostics, planted-temperature checks.
# Reads existing plans only (studies 1-5 and 9; scaled_noise_{r1,a2}_combined_v1_20261031 from scripts/04_scaled_noise.sh).
# CPU, about one hour. Released output: results/robustness_v1_20261102.
source "$(dirname "$0")/common.sh"
$PY_ANALYSIS -u -m frost.analysis.robustness --out $A/robustness_v1_20261102 --boot 2000
