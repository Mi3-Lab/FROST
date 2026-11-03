#!/bin/bash
# Study 14 - exploratory per-clip agreement of decision flips between samplers and the matched reference (400 held-out clips).
# Released output: results/flip_agreement_v1_20261030.
source "$(dirname "$0")/common.sh"
$PY_ANALYSIS -m frost.analysis.flip_agreement --out $A/flip_agreement_v1_20261030
