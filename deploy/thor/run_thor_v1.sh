#!/bin/bash
# FROST Thor v1 runner (protocol protocols/thor.md, registered 29/10/2026 14:59 PDT, section 7).
# Steps, in order, each resumable and each started only after the same "board free" check (gpu_idle):
#   1. P0 bf16 matched-temperature probe (unchanged frost.probes.temperature_probe, clips [0,N), --resume)   -> out/thor_v1/probe_bf16
#   2. latency P0, P1, P2 (thor_latency.py, context_000-009, batch 1 and 6; resumable per clip)      -> out/thor_v1/latency/<V>
#   3. P1 FP8-expert probe  (thor_quant_probe.py)                                                    -> out/thor_v1/P1
#   4. P2 FP8-total probe   (thor_quant_probe.py)                                                    -> out/thor_v1/P2
#   5. P3 NVFP4: layout validation first; probe and latency only if it passed                       -> out/thor_v1/P3
# A step is skipped when its output is complete (manifest.json / latency summary). Never kills or signals anyone
# else's process. Logs to $FROST_THOR_ROOT/out/thor_v1/.
# (When thor_latency.py or thor_quant_probe.py is called without --variant and FROST_THOR_RUNNER is unset, it hands over
# to $FROST_THOR_ROOT/run_thor_v1_next.sh, a copy of this file; this let an already running older runner continue.)
set -u
export FROST_THOR_RUNNER=next
source $FROST_THOR_ROOT/env.sh
OUT=$FROST_THOR_ROOT/out/thor_v1;mkdir -p $OUT;LOG=$OUT/run.log
DATA=$FROST_THOR_ROOT/contexts/physicalai_heldout_contexts_v3_20261027
N_CLIPS=${N_CLIPS:-60}             # fixed by the protocol from the smoke-test time per clip
GPU_BUSY_PCT=${GPU_BUSY_PCT:-10}   # GR3D_FREQ above this counts as busy
MIN_FREE_GB=${MIN_FREE_GB:-32}     # model ~21 GB + activations
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }

exec 9>$OUT/.runner.lock
flock -n 9 || { log "another run_thor_v1 runner holds $OUT/.runner.lock: exiting"; exit 0; }

gpu_idle(){
  # 1) no foreign process holding the GPU apart from the desktop compositor/file manager
  local others=$(for p in $(pgrep -u "$(id -un)"); do
      ls -l /proc/$p/fd 2>/dev/null | grep -q 'nvidia\|nvgpu\|nvmap' && cat /proc/$p/comm; done 2>/dev/null \
      | grep -v -E '^(gnome-shell|mutter-x11-fram|nautilus|Xorg|code|firefox|gnome-control-c)' | sort -u)
  local foreign=$(pgrep -f 'llm_stream_video|joint_filter|trtexec|python.*(train|eval|infer)' | grep -v $$ | head -1)
  # 2) GR3D load over 5 s
  local load=$(timeout 6 tegrastats --interval 1000 2>/dev/null | grep -o 'GR3D_FREQ [0-9]*' | awk '{s+=$2;n++}END{print (n?s/n:100)}')
  local free=$(free -g | awk '/Mem:/{print $7}')
  log "check: gr3d=${load}% avail=${free}GB foreign_pid=${foreign:-none} gpu_procs=[${others//$'\n'/,}]"
  [ -z "$foreign" ] && [ -z "$others" ] && awk -v l=$load -v t=$GPU_BUSY_PCT 'BEGIN{exit !(l<=t)}' && [ "$free" -ge $MIN_FREE_GB ]
}
wait_free(){ log "waiting for a free board: $1"; until gpu_idle; do sleep 300; done; log "board free: $1"; }

log "run_thor_v1 (registered protocol) armed: N_CLIPS=$N_CLIPS pid=$$"
cd $FROST_THOR_ROOT/code/deploy/thor

# 1. P0 bf16 reference grid (unchanged probe; Thor shim selects SDPA attention and the partial offload)
CONFIGS="euler:1:1.0 euler:2:1.0 euler:4:1.0 euler:6:1.0 euler:10:1.0 heun:32:1.0 heun:32:0.9 heun:32:0.8 heun:32:0.7 heun:32:0.6 heun:32:0.5 heun:32:0.4"
if [ -f $OUT/probe_bf16/manifest.json ]; then log "P0 probe complete: skipped"; else
  wait_free "P0 probe"
  python -c "import thor_compat, sys; sys.argv=['temperature_probe']+sys.argv[1:]; from frost.probes.temperature_probe import main; main()" \
    --data $DATA --out $OUT/probe_bf16 --range 0 $N_CLIPS --resume --configs $CONFIGS \
    >> $OUT/probe_bf16.log 2>&1 && log "P0 probe done" || log "P0 probe failed (exit $?)"
fi

# 2. latency P0, P1, P2
for V in P0 P1 P2; do
  if [ -f $OUT/latency/$V/summary.json ] && [ $(ls $OUT/latency/$V/context_0*.json 2>/dev/null | wc -l) -ge 10 ]; then log "latency $V complete: skipped"; continue; fi
  wait_free "latency $V"
  python thor_latency.py --variant $V --out $OUT --data $DATA >> $OUT/latency_$V.log 2>&1 && log "latency $V done" || log "latency $V failed (exit $?)"
done

# 3-4. FP8 variants
for V in P1 P2; do
  if [ -f $OUT/$V/manifest.json ]; then log "$V probe complete: skipped"; continue; fi
  wait_free "$V probe"
  python thor_quant_probe.py --variant $V --out $OUT --data $DATA --range 0 $N_CLIPS >> $OUT/$V.log 2>&1 && log "$V probe done" || log "$V probe failed (exit $?)"
done

# 5. P3 NVFP4 (exploratory), only behind the layout validation
if [ -f $OUT/P3/manifest.json ]; then log "P3 probe complete: skipped"; else
  wait_free "P3 layout validation"
  if python thor_quant_probe.py --variant P3 --out $OUT --data $DATA --validate-only >> $OUT/P3.log 2>&1; then
    log "P3 layout validation passed"
    wait_free "P3 probe"
    python thor_quant_probe.py --variant P3 --out $OUT --data $DATA --range 0 $N_CLIPS >> $OUT/P3.log 2>&1 && log "P3 probe done" || log "P3 probe failed (exit $?)"
  else log "P3 layout validation did not pass (exit $?): P3 not run"; fi
fi
if [ -f $OUT/P3/layout_validation.json ] && grep -q '"passed": true' $OUT/P3/layout_validation.json; then
  if [ -f $OUT/latency/P3/summary.json ] && [ $(ls $OUT/latency/P3/context_0*.json 2>/dev/null | wc -l) -ge 10 ]; then log "latency P3 complete: skipped"; else
    wait_free "latency P3"
    python thor_latency.py --variant P3 --out $OUT --data $DATA >> $OUT/latency_P3.log 2>&1 && log "latency P3 done" || log "latency P3 failed (exit $?)"
  fi
fi
log "run_thor_v1 finished"
