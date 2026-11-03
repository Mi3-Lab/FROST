#!/bin/bash
# Sequential benches/builds on Thor (one GPU job at a time). Usage: run_trt_bench.sh TAG [TAG...]
#   TAG = name of engines/<name>.engine to bench; 'build:<name>:<onnx_dir>:<batch>[:<lmax>]' builds first; 'probe' = precision probe.
set -u
T=$FROST_THOR_ROOT/trt; OUT=$T/out; LOG=$OUT/run_trt_bench.log
source $FROST_THOR_ROOT/env.sh; export PYTHONPATH=$PYTHONPATH:$T/scripts; PY=$T/venv/bin/python; cd $T/scripts
exec 9>$OUT/.trt_proto.lock; flock 9
for tag in "$@"; do
  case $tag in
    probe) $PY trt_precision_probe.py --out $OUT/precision_probe.json > $OUT/precision_probe.log 2>&1; echo "[$(date -u +%FT%TZ)] probe exit $?" >> $LOG;;
    build:*) IFS=: read -r _ n o b l <<< "$tag"; rm -f $T/engines/$n.engine
      $PY build_engine.py --onnx $T/onnx/$o/field.onnx --batch $b --lmax ${l:-4096} --out $T/engines/$n.engine > $OUT/build_$n.log 2>&1
      echo "[$(date -u +%FT%TZ)] build $n exit $?" >> $LOG
      $PY bench_field.py --engine $T/engines/$n.engine --dump $OUT/dump_context_000.pt --out $OUT/bench_$n.json > $OUT/bench_$n.log 2>&1
      echo "[$(date -u +%FT%TZ)] bench $n exit $?" >> $LOG;;
    *) $PY bench_field.py --engine $T/engines/$tag.engine --dump $OUT/dump_context_000.pt --out $OUT/bench_$tag.json > $OUT/bench_$tag.log 2>&1
      echo "[$(date -u +%FT%TZ)] bench $tag exit $?" >> $LOG;;
  esac
done
echo "[$(date -u +%FT%TZ)] done: $*" >> $LOG
