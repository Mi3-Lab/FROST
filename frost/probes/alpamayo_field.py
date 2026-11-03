"""Frozen Alpamayo1.5 action-expert velocity field for one native-format context.

Same capture procedure as native_solver_probe (language seed 123, official
Euler10 replay checked against the model sampler), packaged for reuse.
"""
import copy
import os
import gc
from pathlib import Path
import sys
import numpy as np
import torch

from frost.paths import ALPAMAYO_15, env_path, hf_snapshot

# External locations (configs/paths.example.env).
CHECKPOINT=env_path('FROST_A15_CHECKPOINT',hf_snapshot(*ALPAMAYO_15))


def load_alpamayo():
    sys.path.insert(0,str(env_path('ALPAMAYO15_SRC')/'src'))  # checkout of NVlabs/alpamayo1.5
    from frost.probes.alpamayo_compat import patch_legacy_hydra_targets
    patch_legacy_hydra_targets()
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    model=Alpamayo1_5.from_pretrained(str(CHECKPOINT),dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    model.requires_grad_(False);return model,helper,helper.get_processor(model.tokenizer)


def prepare_field(model,helper,processor,path):
    """Return (field, dims, history_xyz, history_rot, euler10_reference_traj) for a context file."""
    with np.load(path) as z:
        images=torch.from_numpy(z['image_frames']).permute(0,1,4,2,3).contiguous()
        history_xyz=torch.from_numpy(z['ego_history_xyz'])[None,None].cuda()
        history_rot=torch.from_numpy(z['ego_history_rot'])[None,None].cuda()
    assert images.shape[:2]==(4,4) and history_xyz.shape[-2]==16
    messages=helper.create_message(images.flatten(0,1),camera_indices=torch.tensor([0,1,2,6]))
    tokenized=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,continue_final_message=True,
        return_dict=True,return_tensors='pt',min_pixels=196608,max_pixels=196608)
    prepared=helper.to_device({'tokenized_data':tokenized,'ego_history_xyz':history_xyz,'ego_history_rot':history_rot},'cuda')
    captured={};original_sample=model.diffusion.sample
    def capture(*a,**kw):
        fn=kw['step_fn'];values=dict(zip(fn.__code__.co_freevars,[c.cell_contents for c in fn.__closure__]))
        for name in ['prompt_cache','position_ids','attention_mask','forward_kwargs']:captured[name]=copy.deepcopy(values[name])
        torch.manual_seed(9201);torch.cuda.manual_seed_all(9201)
        return original_sample(*a,**kw)
    model.diffusion.sample=capture;torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            reference,_,_=model.sample_trajectories_from_data_with_vlm_rollout(data=prepared,top_p=.98,temperature=.6,num_traj_samples=1,
                diffusion_kwargs={'inference_step':10},max_generation_length=256,return_extra=True)
    finally:model.diffusion.sample=original_sample
    reference=reference.float().cpu().numpy().reshape(-1,3)
    del prepared,tokenized,images;gc.collect();torch.cuda.empty_cache()
    cache=captured['prompt_cache'];prefix=cache.get_seq_length();dims=model.action_space.get_action_space_dims();n=dims[0]
    def field(x,t):
        embeds=model.action_in_proj(x,t)
        if embeds.ndim==2:embeds=embeds.view(len(x),n,-1)
        result=model.expert(inputs_embeds=embeds,position_ids=captured['position_ids'],past_key_values=cache,
            attention_mask=captured['attention_mask'],use_cache=True,**captured['forward_kwargs'])
        cache.crop(prefix)
        return model.action_out_proj(result.last_hidden_state[:,-n:]).view(-1,*dims).float()
    return field,dims,history_xyz,history_rot,reference
