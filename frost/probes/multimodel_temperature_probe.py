"""Replicate the numerical-temperature probe on other frozen flow driving policies.

Our integrators run inside each model's own diffusion.sample call, using the
step_fn (and classifier-free guidance, if the model enables it) that the model
passes there. Plan keys and config names match temperature_probe, so
temperature_report can summarize any model.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from frost.probes.temperature_probe import TAUS

from frost.paths import ALPAMAYO_R1, env_path, hf_snapshot

# External locations (configs/paths.example.env).
CHECKPOINTS={'r1':env_path('FROST_R1_CHECKPOINT',hf_snapshot(*ALPAMAYO_R1))}
CONFIGS=[('euler',k,1.) for k in [1,2,4,6,10]]+[('heun',32,tau) for tau in TAUS]


def integrate(field,initial,steps,method):
    x=initial.clone();h=1/steps
    for i in range(steps):
        t=torch.full((len(x),1,1),i*h,device=x.device);v=field(x,t)
        x=x+h*v if method=='euler' else x+.5*h*(v+field(x+h*v,t+h))
    return x


def make_patched_sample(diffusion,seeds,store):
    original=diffusion.sample;signature=inspect.signature(original)
    def patched(*a,**kw):
        bound=signature.bind_partial(*a,**kw);args=bound.arguments
        step_fn=args['step_fn'];unguided=args.get('unguided_step_fn')
        use_cfg=args.get('use_classifier_free_guidance');use_cfg=getattr(diffusion,'use_classifier_free_guidance',False) if use_cfg is None else use_cfg
        weight=args.get('inference_guidance_weight');weight=getattr(diffusion,'inference_guidance_weight',1.) if weight is None else weight
        def field(x,t):
            v=step_fn(x=x,t=t)
            return (1-weight)*unguided(x=x,t=t)+weight*v if use_cfg else v
        device=args.get('device','cuda')
        for seed in seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=torch.randn(1,*diffusion.x_dims,device=device)
            for method,steps,tau in CONFIGS:
                store[(seed,f'{method}{steps}_t{tau}')]=integrate(field,tau*initial,steps,method).float()
        return original(*a,**kw)
    return original,patched


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['r1','a2'],required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--range',type=int,nargs=2,default=[100,160]);p.add_argument('--seeds',type=int,nargs='+',default=list(range(9201,9207)))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--configs',nargs='+',default=None,help='method:steps:tau triples replacing the default grid (same seeds, same initial noise)')
    args=p.parse_args()
    if args.configs:
        global CONFIGS
        CONFIGS=[(m,int(k),float(t)) for m,k,t in (c.split(':') for c in args.configs)]
    args.out.mkdir(parents=True,exist_ok=args.resume);torch.set_num_threads(2)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    records=sorted((r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r),key=lambda r:r['name'])[args.range[0]:args.range[1]]
    if args.model=='r1':
        sys.path.insert(0,str(env_path('ALPAMAYO_R1_SRC')/'src'))  # checkout of NVlabs/alpamayo (Alpamayo-R1)
        import alpamayo_r1.models.alpamayo_r1 as module
        from alpamayo_r1 import helper
        # This checkout registers diagnostic hooks that copy FFN activations to host RAM; disable them locally.
        for hook in ['register_expert_hooks','register_vision_hooks','register_lm_hooks']:setattr(module,hook,lambda *a,**kw:({},[]))
        model=module.AlpamayoR1.from_pretrained(str(CHECKPOINTS['r1']),dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
        processor=helper.get_processor(model.tokenizer);diffusion=model.diffusion;space=model.action_space
    else:
        sys.path.insert(0,str(env_path('ALPAMAYO2_SRC')/'src'))
        from alpamayo2_super import helper
        from alpamayo2_super.input_profiles import select_task_input
        from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
        model=Alpamayo2Super.from_pretrained('nvidia/Alpamayo2-Super',dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True).eval()
        diffusion=model.diffusion if hasattr(model,'diffusion') else model.expert.diffusion
        space=model.action_space if hasattr(model,'action_space') else model.expert.action_space
    model.requires_grad_(False);plans={}
    if args.resume and (args.out/'plans.npz').exists():
        with np.load(args.out/'plans.npz') as z:plans={k:z[k] for k in z.files}
    done={k.split('_s')[0] for k in plans}
    for record in records:
        if record['name'] in done:continue
        name=record['name'];store={};original,patched=make_patched_sample(diffusion,args.seeds,store)
        random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        diffusion.sample=patched
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                if args.model=='r1':
                    with np.load(args.data/f'{name}.npz') as z:
                        frames=torch.from_numpy(z['image_frames']).permute(0,1,4,2,3).flatten(0,1).contiguous()
                        hx=torch.from_numpy(z['ego_history_xyz'])[None,None].cuda();hr=torch.from_numpy(z['ego_history_rot'])[None,None].cuda()
                    tokens=processor.apply_chat_template(helper.create_message(frames),tokenize=True,add_generation_prompt=False,
                        continue_final_message=True,return_dict=True,return_tensors='pt')
                    prepared=helper.to_device({'tokenized_data':tokens,'ego_history_xyz':hx,'ego_history_rot':hr},'cuda')
                    model.sample_trajectories_from_data_with_vlm_rollout(data=prepared,top_p=.98,temperature=.6,num_traj_samples=1,
                        diffusion_kwargs={'inference_step':10},max_generation_length=256)
                else:
                    data=select_task_input(load_physical_aiavdataset(record['clip_id'],t0_us=record['t0_us']),'trajectory')
                    hx=data['ego_history_xyz'].cuda();hr=data['ego_history_rot'].cuda()
                    inputs=helper.to_device(helper.prepare_model_inputs(data,model.config,model.tokenizer),'cuda')
                    model.sample_trajectories_from_data(data=inputs,top_p=.98,temperature=.6,num_traj_samples=1,
                        diffusion_kwargs={'inference_step':10})
                for (seed,cfg),action in store.items():
                    xyz,_=space.action_to_traj(action,hx[:,0].float(),hr[:,0].float())
                    plans[f'{name}_s{seed}_{cfg}']=xyz.float().cpu().numpy().reshape(-1,3)
        finally:diffusion.sample=original
        np.savez_compressed(args.out/'plans.npz',**plans);print(json.dumps({'context':name,'plans':len(plans)}),flush=True)
        torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'model':args.model,'data':str(args.data),'range':args.range,
        'seeds':args.seeds,'configs':CONFIGS,'source_sha256':hashlib.sha256(source).hexdigest(),
        'limitations':['Language branch sampled once per clip (seed 123)','Alpamayo2 uses its own 7-camera release input profile streamed from HF',
            'R1 fed the same 4-camera x 4-frame cache as Alpamayo1.5']},indent=2))


if __name__=='__main__':main()
