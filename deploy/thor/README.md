# FROST on NVIDIA DRIVE AGX Thor

Scripts behind the paper's deployment numbers (protocol: `protocols/thor.md`, study 12). Everything here runs **on the
device** and writes plans, latencies and fidelity files; the statistics are computed afterwards on a workstation with
`scripts/12_thor.sh` (`frost.analysis.thor_analysis`, `thor_addendum3_analysis`, `thor_precision_flips`).

**Scope of the latency numbers.** The reported latency is that of the Alpamayo-1.5 *action expert* (the flow-matching
sampler) per call and per plan. It is not the end-to-end latency of the full VLA (vision encoder, reasoning decode and
expert), except where a table says so explicitly (compiled-VLM benchmark).

## Software on the device

| Component | Version used |
|---|---|
| DriveOS | 7.0.3 (about 14.7 GiB visible to CUDA) |
| CUDA | 12.8 |
| PyTorch | 2.8.0 (Thor build) |
| TensorRT | 10.10.0 |
| NVIDIA ModelOpt | 0.47.0 (FP8 / NVFP4 quantization and ONNX export) |

The device runs offline. Copy the checkpoint (`nvidia/Alpamayo-1.5-10B`, revision in `THIRD_PARTY.md`), the
Hugging Face config/tokenizer/processor files of `nvidia/Cosmos-Reason2-8B` and `Qwen/Qwen3-VL-2B-Instruct`, the
held-out context cache (`physicalai_heldout_contexts_v3_20261027`, see `scripts/00_data.sh`) and, for Qwen-Drive,
`Qwen/Qwen-Drive-1.0-4B`, from a workstation. No token is used on the device.

## Layout

```
$FROST_THOR_ROOT/                 (default ./thor_work)
  env.sh                          copy of deploy/thor/env.sh (sourced by every runner)
  venv/                           PyTorch environment;  trt/venv/  TensorRT + ModelOpt environment
  code/                           checkout of this repository (frost package + deploy/thor)
  alpamayo1.5/                    NVlabs/alpamayo1.5 at the commit in THIRD_PARTY.md
  models/Alpamayo-1.5-10B/        checkpoint;  models/Qwen-Drive-1.0-4B/
  contexts/physicalai_heldout_contexts_v3_20261027/
  trt/scripts/                    copy of deploy/thor/trt/*;   vlm/scripts/  copy of deploy/thor/vlm/*
  out/thor_v1/  trt/{onnx,engines,prefix,frost400,out}/  vlm/{onnx,engines,frost400}/  qd/{scenes,engines,out}/
```

```bash
export FROST_THOR_ROOT=$PWD/thor_work
cp deploy/thor/env.sh $FROST_THOR_ROOT/env.sh
mkdir -p $FROST_THOR_ROOT/trt/scripts $FROST_THOR_ROOT/vlm/scripts
cp deploy/thor/trt/* $FROST_THOR_ROOT/trt/scripts/; cp deploy/thor/vlm/* $FROST_THOR_ROOT/vlm/scripts/
source $FROST_THOR_ROOT/env.sh
```

The board was shared: `run_thor_v1.sh` and `trt/thor_guard.py` only start a GPU step when no other process uses the
GPU (they never signal other processes). Set `FROST_TRT_NO_GUARD=1` on a dedicated board.

## 1. Protocol v1: PyTorch on the device (60 clips) -> `thor_v1_20261030`

`thor_compat.py` is imported before the unchanged `frost.probes.temperature_probe`: it selects SDPA attention and
streams the language layers that do not fit in device memory. Variants: P0 bf16, P1 FP8 action expert, P2 FP8 expert
and language model, P3 NVFP4 (exploratory, only after its layout validation passes).

```bash
bash deploy/thor/run_thor_v1.sh            # N_CLIPS=60; resumable; steps:
#   P0 probe : python -c "import thor_compat, sys; sys.argv=['temperature_probe']+sys.argv[1:]; from frost.probes.temperature_probe import main; main()" \
#              --data $DATA --out $OUT/probe_bf16 --range 0 60 --resume --configs euler:1:1.0 euler:2:1.0 euler:4:1.0 euler:6:1.0 euler:10:1.0 \
#              heun:32:1.0 heun:32:0.9 heun:32:0.8 heun:32:0.7 heun:32:0.6 heun:32:0.5 heun:32:0.4
#   latency  : python thor_latency.py --variant {P0,P1,P2} --out $OUT --data $DATA      (context_000-009, batch 1 and 6)
#   P1, P2   : python thor_quant_probe.py --variant {P1,P2} --out $OUT --data $DATA --range 0 60
```

Outputs: `out/thor_v1/{probe_bf16,P1,P2}/plans.npz`, `out/thor_v1/latency/<V>/summary.json` (the paper reads
`latency/P0/summary.json`). Copy `out/thor_v1` to `$FROST_ARTIFACTS/thor_v1_20261030`.

## 2. TensorRT action expert (addendum 2) -> `thor_trt_bench_20261031`, `thor_trt_frost400_20261031`

1. Export the expert field to ONNX on a workstation or the device (`trt/export_onnx.py`, module in `trt/field_trt.py`;
   `trt/onnx_pack.py` packs external weights). FP8 / NVFP4 graphs come from ModelOpt with calibration on development
   clips that are not evaluation clips: `trt/quantize_export.py` (driven by `trt/run_trt_final_calib.sh`;
   `trt/nvfp4_fix.py` patches the NVFP4 export).
2. Build engines: `python trt/build_engine.py --onnx <onnx_dir>/field.onnx --batch <B> --lmax <L> --out engines/<name>.engine`.
3. Reference inputs for the bench: `python trt/dump_clip.py --clip context_000 --out $T/out/dump_context_000.pt`.
4. Bench (batch 1, 3200-token prefix, CUDA graphs): `bash trt/run_trt_bench.sh bf16_b1_L3200 fp8tv2_b1_L3200 nvfp4ne_b1_L3200`
   (`build:<name>:<onnx_dir>:<batch>[:<lmax>]` builds first; `probe` runs `trt/trt_precision_probe.py`).
   Outputs `trt/out/bench_<name>.json`; the paper reads the three files above.
5. FROST on 400 clips with the batch-6 engines: `bash trt/run_trt_frost400.sh`, i.e.
   `thor_trt_probe.py prefix --range 0 400 --resume --out $R` (PyTorch bf16 prefix once per clip), then
   `thor_trt_probe.py field --variant {trt_bf16,trt_fp8,trt_nvfp4} --engine engines/probe_{bf16,fp8,nvfp4}_b6_L3456.engine --range 0 400 --resume --out $R`.
   Copy `$R` to `$FROST_ARTIFACTS/thor_trt_frost400_20261031`.

## 3. Compiled reasoning VLM (addendum 3) -> `thor_vlm_trt_20261031`, `thor_vlm_frost400_20261101`

`vlm/vlm_graph.py` defines export-friendly vision / text-chunk / head graphs, `vlm/export_vlm.py` exports and quantizes
them (`bf16`, `calibdata`, `quant --precision fp8|nvfp4`), `vlm/attn_surgery.py` rewrites attention for TensorRT,
`vlm/build_vlm.py` builds the engines and `vlm/vlm_runtime.py` / `vlm/vlm_trt.py` run them.

```bash
python vlm/bench_vlm.py --engines vlm/engines/fp8_s1 --field trt/engines/fp8tv2_b1_L3456.engine --range 0 5 --repeats 3 \
    --preload --gpu-preproc --out vlm/out/bench_fp8_s1_gpupre.json                     # full-VLA timing (paper)
python vlm/validate_vlm.py --backend trt --engines vlm/engines/<recipe> --range 0 40 --field trt/engines/probe_bf16_b6_L3456.engine --field-ref trt/frost400/trt_bf16/plans.npz --out vlm/out/val_<recipe>.json   # fidelity
python vlm/analyze_vlm_val.py --val val40_bf16.json val40_fp8.json val40_nvfp4.json --ref ref_plans_bf16.npz ref_plans_fp8.npz ref_plans_nvfp4.npz --out val40_summary.json
FLD=trt/engines/probe_fp8_b6_L3456.engine
python vlm/thor_vlm_frost.py --variant comp_fp8 --engines vlm/engines/fp8_s1 --field $FLD --range 0 400 --out vlm/frost400
python vlm/thor_vlm_frost.py --variant comp_fp8pre_awqfulldec --engines vlm/engines/fp8_s1 --decode-engines vlm/engines/nvfp4awqfull \
    --field $FLD --range 0 400 --out vlm/frost400
```

Copy `vlm/out` to `$FROST_ARTIFACTS/thor_vlm_trt_20261031/out` and `vlm/frost400` to
`$FROST_ARTIFACTS/thor_vlm_frost400_20261101`.

## 4. Qwen-Drive on the device -> `thor_qwendrive_20261101`

```bash
python deploy/thor/qwendrive/make_scenes.py --records <contexts>/records.json --out qd/scenes --range 0 40      # workstation
python deploy/thor/qwendrive/export_expert.py --model $FROST_QWENDRIVE_MODEL --out qd/onnx_expert              # workstation
python deploy/thor/qwendrive/build_engine.py --onnx qd/onnx_expert/expert.onnx --out qd/engines/expert.engine   # device
python deploy/thor/qwendrive/eager_bench.py --model models/Qwen-Drive-1.0-4B --scenes qd/scenes --range 0 40 --out qd/out/eager40
python deploy/thor/qwendrive/trt_bench.py --model models/Qwen-Drive-1.0-4B --scenes qd/scenes --engine qd/engines/<expert> \
    [--fp32-prep] [--graph-decode] [--vision-engines <dir>] --range 0 40 --out qd/out/trt40_<variant>
python deploy/thor/qwendrive/analyze_fidelity.py --run qd/out/trt40_<variant> --ref <workstation fp32 reference plans> --n 40   # -> fidelity.json
```

The paper reads `eager40/fidelity.json`, `trt40_bf16/fidelity.json` and `trt40_fp32_gd/fidelity.json`.
`qwendrive/probe_v3_bf16.py` (`run_probe_v3_bf16.sh`) runs the registered v3 Qwen-Drive probe on the device.
`thor_patches.py`, `expert_graph.py`, `expert_trt.py`, `graph_decode.py` and `vision_trt.py` are imported by these scripts.

Arguments shown as `...` or `<...>` varied between repeated launches on the shared board; the exact values of each run
are stored in the JSON outputs (`results/thor_*`).
