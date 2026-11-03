#!/bin/bash
# Regenerate every table, macro file and figure of the paper from results/ and check the tables against the published
# SHA-256 digests (CPU, under a minute). Outputs: reproduce/output/{tables,figures}.
source "$(dirname "$0")/common.sh"
$PY_ANALYSIS reproduce/make_all.py --check
