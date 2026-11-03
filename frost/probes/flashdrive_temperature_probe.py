"""Matched-temperature probe for FlashDrive (z-lab) and the original Alpamayo 1.5 checkpoint.

Run with the isolated FlashDrive venv. Inputs come from FlashDrive's own
scripts/infer.py (create_inputs): consecutive 10 Hz windows per PhysicalAI-AV clip,
streaming form for the z-lab stack. Inside each window's action sampling we take the
step_fn the model provides and integrate every configuration from the same VLM
reasoning, with six fixed noise seeds: FlashDrive's cached Euler (8 steps, cache
3-6), plain Euler 8/10/4, and Heun32 at several initial-noise scales tau. The model
then continues with its own official result. Eager mode (torch_compile=None).
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch

CONFIGS=[('flashcache',8,1.),('euler',8,1.),('euler',10,1.),('euler',4,1.)]+[('heun',32,t) for t in [1.,.9,.8,.7,.6,.5]]
CACHE_STEPS=[3,4,5,6];SEEDS=list(range(9201,9207))


def integrate(step_fn,noise,steps,method):
    x=noise;dims=[1]*(noise.dim()-1);h=1./steps;v=None
    for i in range(steps):
        t=torch.full((x.shape[0],*dims),i*h,device=x.device,dtype=x.dtype)
        if method=='flashcache' and i in CACHE_STEPS:pass
        else:v=step_fn(x=x,t=t)
        if method=='heun':x=x+.5*h*(v+step_fn(x=x+h*v,t=t+h))
        else:x=x+h*v
    return x


def run_configs(step_fn,x_dims,device,store):
    for seed in SEEDS:
        generator=torch.Generator(device=device).manual_seed(seed)
        base=torch.randn(1,*x_dims,device=device,generator=generator)
        for method,steps,tau in CONFIGS:
            store[(seed,f'{method}{steps}_t{tau}')]=integrate(step_fn,tau*base,steps,method).float()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--records',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--range',type=int,nargs=2,default=[100,160])
    p.add_argument('--windows',type=int,default=4,help='Windows per clip; the first only prefills on the streaming stack')
    p.add_argument('--flashdrive',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=args.resume)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    import flashdrive
    import flashdrive._base as fd_base
    spec=importlib.util.spec_from_file_location('fd_infer',args.flashdrive/'scripts/infer.py');infer=importlib.util.module_from_spec(spec);spec.loader.exec_module(infer)
    helper=infer.load_helper(args.model);model=flashdrive.from_pretrained(args.model,device='cuda',torch_compile=None)
    optimized=isinstance(model,flashdrive.FlashDriveMixin);processor=helper.get_processor(model.tokenizer)
    diffusion=model.diffusion;store={}
    if optimized:
        original=fd_base.euler_with_cache
        def patched(diffusion_,*,noise,batch_size,step_fn,cache_steps,device,inference_step):
            run_configs(step_fn,diffusion_.x_dims,device,store)
            return original(diffusion_,noise=noise,batch_size=batch_size,step_fn=step_fn,cache_steps=cache_steps,device=device,inference_step=inference_step)
        fd_base.euler_with_cache=patched
        rollout=lambda data:model.sample_trajectories_streaming(data=data,num_traj_samples=1,return_extra=True,max_new_tokens=infer.MAX_NEW_TOKENS,
            diffusion_kwargs={'inference_step':infer.DIFFUSION_STEPS,'cache_steps':infer.CACHE_STEPS,'int_method':'euler_with_cache'})
    else:
        original=diffusion.sample;signature=inspect.signature(original)
        def patched(*a,**kw):
            bound=signature.bind_partial(*a,**kw).arguments
            run_configs(bound['step_fn'],diffusion.x_dims,bound.get('device','cuda'),store)
            return original(*a,**kw)
        diffusion.sample=patched
        rollout=lambda data:model.sample_trajectories_from_data_with_vlm_rollout(data=data,num_traj_samples=1,return_extra=True)
    records=sorted((r for r in json.loads(args.records.read_text()) if 'error' not in r),key=lambda r:r['name'])[args.range[0]:args.range[1]]
    plans={}
    if args.resume and (args.out/'plans.npz').exists():
        with np.load(args.out/'plans.npz') as z:plans={k:z[k] for k in z.files}
    done={k.split('_w')[0] for k in plans}
    for record in records:
        name=record['name']
        if name in done:continue
        if optimized:
            # Each clip is a fresh stream: the streaming KV cache and the shape-bound primitive buffers belong to the previous clip.
            model._past_key_values=None;model.setup_torch_compile(None);model.setup_streaming(extended_vision_ranges=model._extended_vision_ranges)
            for module in model.modules():
                if hasattr(module,'reset_shape_caches'):module.reset_shape_caches()
        # Windows end at the clip's t0 so the last one matches the open-loop studies.
        start=record['t0_us']-(args.windows-1)*100_000
        inputs=infer.create_inputs(helper,processor,args.windows,record['clip_id'],start,100_000,streaming=optimized)
        for w,model_inputs in enumerate(inputs):
            store.clear();data=helper.to_device(model_inputs,'cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred,_,_=rollout(data)
                if pred is None or not store:continue
                hx=data['ego_history_xyz'][:,0].float();hr=data['ego_history_rot'][:,0].float()
                for (seed,cfg),action in store.items():
                    xyz,_=model.action_space.action_to_traj(action,hx,hr);plans[f'{name}_w{w}_s{seed}_{cfg}']=xyz.float().cpu().numpy().reshape(-1,3)
            plans[f'{name}_w{w}_human']=model_inputs['ego_future_xyz'][0,0].float().numpy()
            plans[f'{name}_w{w}_official']=pred.float().cpu().numpy().reshape(-1,3)
        np.savez_compressed(args.out/'plans.npz',**plans);print(json.dumps({'context':name,'plans':len(plans)}),flush=True)
        torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'model':args.model,'optimized_stack':optimized,'range':args.range,
        'windows':args.windows,'seeds':SEEDS,'configs':CONFIGS,'cache_steps':CACHE_STEPS,'source_sha256':hashlib.sha256(source).hexdigest()},indent=2))


if __name__=='__main__':main()
