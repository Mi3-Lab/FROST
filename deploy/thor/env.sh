# FROST on NVIDIA DRIVE AGX Thor: source this before any run on the device.
# Layout under $FROST_THOR_ROOT (default ./thor_work): venv/ (JetPack/DriveOS PyTorch + TensorRT environment),
# code/ (a checkout of this repository), alpamayo1.5/ (NVlabs/alpamayo1.5), models/Alpamayo-1.5-10B (checkpoint copied
# from a workstation), contexts/ (the held-out context cache), hf/ (Hugging Face cache with the config, tokenizer and
# processor files of nvidia/Cosmos-Reason2-8B and Qwen/Qwen3-VL-2B-Instruct; no weights), out/, trt/, vlm/, qd/.
# The device runs offline: copy models and data from a workstation; no Hugging Face token is needed or used.
export FROST_THOR_ROOT=${FROST_THOR_ROOT:-$PWD/thor_work}
source $FROST_THOR_ROOT/venv/bin/activate
export FROST_A15_CHECKPOINT=$FROST_THOR_ROOT/models/Alpamayo-1.5-10B
export ALPAMAYO15_SRC=$FROST_THOR_ROOT/alpamayo1.5
export PYTHONPATH=$FROST_THOR_ROOT/code:$FROST_THOR_ROOT/code/deploy/thor
export HF_HOME=$FROST_THOR_ROOT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=$FROST_THOR_ROOT/tmp PIP_CACHE_DIR=$FROST_THOR_ROOT/pipcache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# DriveOS 7.0.3 exposes about 14.7 GiB to CUDA: the language layers beyond FROST_THOR_GPU_LAYERS are streamed (thor_compat.py).
export FROST_THOR_OFFLOAD=1 FROST_THOR_GPU_LAYERS=999 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
