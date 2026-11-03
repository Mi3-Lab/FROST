#!/bin/bash
# Thor: calibration for the probe engines on 8 clips of the dev set physicalai_test_contexts_v2 (disjoint from the 400
# held-out evaluation clips), then the TRT probe's prefix phase for the 2-clip smoke test. Serialised by the shared lock.
set -u
T=$FROST_THOR_ROOT/trt; OUT=$T/out; source $FROST_THOR_ROOT/env.sh
PYDEV=${PYDEV:-$FROST_THOR_ROOT/build/pydev/root/usr/include}  # Python 3.12 headers for the ModelOpt CUDA extensions; export CPATH=$PYDEV/python3.12:$PYDEV/aarch64-linux-gnu/python3.12:$PYDEV
export PYTHONPATH=$PYTHONPATH:$T/scripts PATH=$T/venv/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
cd $T/scripts; exec 9>$OUT/.trt_proto.lock; flock 9
L=$OUT/run_trt_bench.log; C=$OUT/calib_testv2
python dump_clip.py --data $T/calib_data_testv2 --out $C --calib-clips $(cd $T/calib_data_testv2; ls *.npz | sed 's/.npz//') > $OUT/calib_testv2_dump.log 2>&1
echo "[$(date -u +%FT%TZ)] testv2 calib dumps exit $?" >> $L
python quantize_export.py --mode fp8 --calib $C/*.pt --out $OUT/q_fp8_testv2 --save-amax $OUT/amax_fp8_testv2.pt --calib-only > $OUT/calib_fp8_testv2.log 2>&1
echo "[$(date -u +%FT%TZ)] calib fp8 testv2 exit $?" >> $L
python quantize_export.py --mode nvfp4 --disable '*layers.0.*' '*layers.1.*' '*layers.34.*' '*layers.35.*' --calib $C/*.pt --out $OUT/q_nvfp4ne_testv2 --save-amax $OUT/amax_nvfp4ne_testv2.pt --calib-only > $OUT/calib_nvfp4ne_testv2.log 2>&1
echo "[$(date -u +%FT%TZ)] calib nvfp4 no_ends testv2 exit $?" >> $L
python thor_trt_probe.py prefix --range 0 2 --resume > $OUT/probe_prefix_smoke.log 2>&1
echo "[$(date -u +%FT%TZ)] probe prefix smoke exit $?" >> $L
