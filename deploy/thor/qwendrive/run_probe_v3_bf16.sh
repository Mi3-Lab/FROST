#!/bin/bash
# Resumable launcher for probe_v3_bf16.py on the Thor (restarts after a crash; finished clips are skipped via OUT/parts/<name>.npz).
source $FROST_THOR_ROOT/code/deploy/thor/env.sh; export PYTHONPATH=$PYTHONPATH:$QWENDRIVE_SRC/src
cd $FROST_THOR_ROOT/code/deploy/thor/qwendrive
OUT=$FROST_THOR_ROOT/qd/out/probe_v3_bf16_thor
mkdir -p $OUT
for try in 1 2 3 4 5; do
  python probe_v3_bf16.py --model $FROST_THOR_ROOT/models/Qwen-Drive-1.0-4B --scenes $FROST_THOR_ROOT/qd/scenes \
    --out $OUT --range 0 400 --wait-scenes >> $OUT.log 2>&1 && break
  echo "RESTART $try $(date)" >> $OUT.log; sleep 30
done
