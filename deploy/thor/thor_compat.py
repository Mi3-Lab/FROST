"""Thor-only compatibility shim (not part of the workstation code).

The released Alpamayo-1.5 config asks for attn_implementation='flash_attention_2'. On the
workstation transformers 4.57 falls back to the hub kernel kernels-community/flash-attn
(x86_64 build only). No flash-attn build exists for aarch64 / sm_101 offline, so on Thor we
map every flash_attention_* request to PyTorch SDPA. This is a declared deviation: the
attention backend differs from the workstation run (FA2 hub kernel vs SDPA).
"""
import transformers.modeling_utils as mu

_orig=mu.PreTrainedModel._check_and_adjust_attn_implementation


def _patched(self,attn_implementation,is_init_check=False):
    if isinstance(attn_implementation,str) and attn_implementation.startswith('flash_attention'):
        attn_implementation='sdpa'
    return _orig(self,attn_implementation,is_init_check)


mu.PreTrainedModel._check_and_adjust_attn_implementation=_patched


# ---------------------------------------------------------------------------------------------
# GPU memory: DriveOS 7.0.3 exposes only ~14.7 GiB to CUDA (cudaGetDeviceProperties: 15020 MB),
# although the board has 58 GB. Alpamayo-1.5 in bf16 is 20.6 GiB (language model 14.1, expert
# 4.25, lm_head 1.19, vision 1.07). load_alpamayo_thor keeps the vision tower, embeddings,
# lm_head, the action expert and the first FROST_THOR_GPU_LAYERS language layers resident on the
# GPU and lets accelerate stream the remaining language layers from host memory (executed on
# the GPU, same bf16 arithmetic). The diffusion field only calls the action expert on the cached
# prefix, so per-call latency of the field is not affected by the offload; the VLM prefill and
# reasoning rollout are (slower than a resident deployment).
import os


def load_alpamayo_thor():
    import sys
    from pathlib import Path
    import torch
    from accelerate import init_empty_weights
    import frost.probes.alpamayo_field as af
    sys.path.insert(0,str(Path(os.environ['ALPAMAYO15_SRC'])/'src'))  # checkout of NVlabs/alpamayo1.5
    from frost.probes.alpamayo_compat import patch_legacy_hydra_targets
    patch_legacy_hydra_targets()
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5 import helper
    gpu_layers=int(os.environ.get('FROST_THOR_GPU_LAYERS','12'))
    config=Alpamayo1_5Config.from_pretrained(str(af.CHECKPOINT),local_files_only=True)
    with init_empty_weights():skeleton=Alpamayo1_5(config)
    dm={}
    for name,child in skeleton.named_children():
        if name!='vlm':dm[name]=0;continue
        for n2,c2 in child.named_children():
            if n2!='model':dm[f'vlm.{n2}']=0;continue
            for n3,c3 in c2.named_children():
                if n3!='language_model':dm[f'vlm.model.{n3}']=0;continue
                for n4,c4 in c3.named_children():
                    if n4!='layers':dm[f'vlm.model.language_model.{n4}']=0;continue
                    for i in range(len(c4)):dm[f'vlm.model.language_model.layers.{i}']=0 if i<gpu_layers else 'cpu'
    del skeleton
    model=Alpamayo1_5.from_pretrained(str(af.CHECKPOINT),dtype=torch.bfloat16,local_files_only=True,device_map=dm).eval()
    model.requires_grad_(False);model._thor_device_map={'gpu_layers':gpu_layers,'n_offloaded':sum(v=='cpu' for v in dm.values())}
    return model,helper,helper.get_processor(model.tokenizer)


if os.environ.get('FROST_THOR_OFFLOAD','1')=='1':
    import frost.probes.alpamayo_field as _af
    _af.load_alpamayo=load_alpamayo_thor
