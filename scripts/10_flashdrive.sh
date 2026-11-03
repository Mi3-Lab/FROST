#!/bin/bash
# Study 10 - FlashDrive (velocity caching, z-lab) on the original 60 test clips: NVIDIA Alpamayo-1.5 checkpoint and the
# z-lab released stack, four 10 Hz windows per clip. Environment: the FlashDrive venv; $FLASHDRIVE_SRC checkout.
# Released outputs: results/flashdrive_analysis_{nvidia_v2_20261026,zlab_v1_20261027}, results/flashdrive_flips_v1_20261028.
source "$(dirname "$0")/common.sh"
T=$D/physicalai_test_contexts_v1_20261022/records.json
if stage probe; then
  for m in nvidia z-lab; do tag=${m/-/}
    $PY_FLASHDRIVE -P -m frost.probes.flashdrive_temperature_probe --model $m/Alpamayo-1.5-10B --records $T \
      --out $A/flashdrive_probe_${tag}_v1 --range 100 160 --windows 4 --flashdrive $FLASHDRIVE_SRC --resume
  done
fi
if stage analysis; then
  $PY_ANALYSIS -m frost.analysis.flashdrive_analysis --probe $A/flashdrive_probe_nvidia_v1 --out $A/flashdrive_analysis_nvidia_v2_20261026
  $PY_ANALYSIS -m frost.analysis.flashdrive_analysis --probe $A/flashdrive_probe_zlab_v1 --out $A/flashdrive_analysis_zlab_v1_20261027
  $PY_ANALYSIS -m frost.analysis.flashdrive_flips --probe $A/flashdrive_probe_nvidia_v1 $A/flashdrive_probe_zlab_v1 --records $T \
    --out $A/flashdrive_flips_v1_20261028
fi
