"""Native-format synthetic CARLA gate for frozen Alpamayo1.5 numerical solvers."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def integrate(field,initial,steps,method):
    x=initial.clone();states=[];velocities=[]
    times=torch.linspace(0.,1.,steps+1,device=x.device,dtype=x.dtype)
    if method=='early_euler':times=times.square()
    if method=='cosine_euler':times=(1-torch.cos(np.pi*times))/2
    for i in range(steps):
        t=times[i].view(1,1,1).expand(len(x),1,1)
        dt=(times[i+1]-times[i]).view(1,1,1).expand(len(x),1,1)
        # FlashDrive-style: fresh first two and last two steps, middle reuses the last fresh velocity.
        cached=method=='cached_euler' and 2<=i<steps-2
        v=v if cached else field(x,t);states.append(x.detach().cpu().numpy());velocities.append(v.detach().cpu().numpy())
        if method=='heun':
            v2=field(x+dt*v,t+dt);x=x+.5*dt*(v+v2)
        elif method=='rk4':
            v2=field(x+.5*dt*v,t+.5*dt);v3=field(x+.5*dt*v2,t+.5*dt)
            v4=field(x+dt*v3,t+dt);x=x+dt*(v+2*v2+2*v3+v4)/6
        elif method in ['euler','early_euler','cosine_euler','cached_euler']:x=x+dt*v
        else:raise ValueError(method)
    return x,np.stack(states),np.stack(velocities)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--euler-steps',type=int,nargs='+',default=[4,8,10,20,40,80])
    p.add_argument('--heun-steps',type=int,nargs='+',default=[4,10,20,40])
    p.add_argument('--action-seeds',type=int,nargs='+',default=[9201,9202])
    p.add_argument('--precision',choices=['bf16','expert_fp32'],default='bf16')
    p.add_argument('--control-solvers',action='store_true')
    p.add_argument('--protocol',type=Path,default=Path('protocols/decision_flip_gate.md'))
    p.add_argument('--solvers',nargs='+',default=None,help='Explicit method:steps list such as euler:10 heun:160 cached_euler:8; overrides step lists')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    protocol=args.protocol.read_bytes();(args.out/'protocol_snapshot.md').write_bytes(protocol)
    from frost.paths import env_path
    from frost.probes.alpamayo_field import CHECKPOINT
    sys.path.insert(0,str(env_path('ALPAMAYO15_SRC')/'src'))  # checkout of NVlabs/alpamayo1.5
    from frost.probes.alpamayo_compat import patch_legacy_hydra_targets
    patch_legacy_hydra_targets()
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    checkpoint=CHECKPOINT
    print('Loading cached Alpamayo1.5',flush=True)
    model=Alpamayo1_5.from_pretrained(str(checkpoint),dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    model.requires_grad_(False);processor=helper.get_processor(model.tokenizer);original_sample=model.diffusion.sample
    records=[];outputs={};source_model=Path(sys.modules[Alpamayo1_5.__module__].__file__)
    paths=sorted(args.data.glob('context_*.npz'))
    if not paths:raise RuntimeError('No collected contexts')
    for path in paths:
        with np.load(path) as z:
            images=torch.from_numpy(z['image_frames']).permute(0,1,4,2,3).contiguous()
            # Recorder stores [camera,time,height,width,RGB].
            history_xyz=torch.from_numpy(z['ego_history_xyz'])[None,None].cuda()
            history_rot=torch.from_numpy(z['ego_history_rot'])[None,None].cuda()
        assert images.shape[:2]==(4,4) and history_xyz.shape[-2]==16
        messages=helper.create_message(images.flatten(0,1),camera_indices=torch.tensor([0,1,2,6]))
        tokenized=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,continue_final_message=True,
            return_dict=True,return_tensors='pt',min_pixels=196608,max_pixels=196608)
        prepared=helper.to_device({'tokenized_data':tokenized,'ego_history_xyz':history_xyz,'ego_history_rot':history_rot},'cuda')
        captured={}
        def capture(*a,**kw):
            fn=kw['step_fn'];values=dict(zip(fn.__code__.co_freevars,[c.cell_contents for c in fn.__closure__]))
            for name in ['prompt_cache','position_ids','attention_mask','forward_kwargs']:captured[name]=copy.deepcopy(values[name])
            torch.manual_seed(9201);torch.cuda.manual_seed_all(9201)
            return original_sample(*a,**kw)
        model.diffusion.sample=capture;torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            reference,_,_=model.sample_trajectories_from_data_with_vlm_rollout(data=prepared,top_p=.98,temperature=.6,num_traj_samples=1,
                diffusion_kwargs={'inference_step':10},max_generation_length=256,return_extra=True)
        reference=reference.float().cpu().numpy().reshape(-1,3);model.diffusion.sample=original_sample
        del prepared,tokenized,images;gc.collect();torch.cuda.empty_cache()
        cache=captured['prompt_cache'];prefix=cache.get_seq_length();dims=model.action_space.get_action_space_dims();n=dims[0]
        if args.precision=='expert_fp32':
            model.expert.float();model.action_in_proj.float();model.action_out_proj.float()
            for layer in cache.layers:layer.keys=layer.keys.float();layer.values=layer.values.float()
            captured['attention_mask']=captured['attention_mask'].float()
        def field(x,t):
            embeds=model.action_in_proj(x,t)
            if embeds.ndim==2:embeds=embeds.view(len(x),n,-1)
            result=model.expert(inputs_embeds=embeds,position_ids=captured['position_ids'],past_key_values=cache,
                attention_mask=captured['attention_mask'],use_cache=True,**captured['forward_kwargs'])
            cache.crop(prefix)
            return model.action_out_proj(result.last_hidden_state[:,-n:]).view(-1,*dims).float()
        for seed in args.action_seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=torch.randn(1,*dims,device='cuda')
            solvers=[('euler',n) for n in args.euler_steps]+[('heun',n) for n in args.heun_steps]
            if args.solvers:
                solvers=[(s.split(':')[0],int(s.split(':')[1])) for s in args.solvers]
            if args.control_solvers:
                solvers += [(method,n) for method in ['early_euler','cosine_euler'] for n in [4,8,10]]
                solvers += [('rk4',n) for n in [1,2,5,10]]
            for method,steps in solvers:
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.precision=='bf16'):
                    action,states,velocities=integrate(field,initial,steps,method)
                    xyz,_=model.action_space.action_to_traj(action,history_xyz[:,0],history_rot[:,0])
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                trajectory=xyz.float().cpu().numpy().reshape(-1,3);key=f'{path.stem}_s{seed}_{method}{steps}'
                outputs[key]=trajectory;outputs[key+'_action']=action.cpu().numpy()
                if method=='euler' and steps==10:outputs[key+'_states']=states;outputs[key+'_velocities']=velocities
                replay=float(np.linalg.norm(trajectory[:,:2]-reference[:,:2],axis=-1).mean()) if seed==9201 and method=='euler' and steps==10 and args.precision=='bf16' else None
                if replay is not None and replay>1e-4:raise RuntimeError(f'Native cache replay mismatch: {replay}')
                record={'context':path.stem,'seed':seed,'method':method,'steps':steps,'field_evaluations':min(steps,4) if method=='cached_euler' else steps*({'heun':2,'rk4':4}.get(method,1)),
                    'seconds':elapsed,'replay_error_m':replay,'finite':bool(np.isfinite(trajectory).all()),'cache_tokens':prefix}
                records.append(record);(args.out/'records.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
            np.savez_compressed(args.out/'predictions.npz',**outputs)
        del captured,cache;gc.collect();torch.cuda.empty_cache()
    manifest={'scope':__doc__,'data':str(args.data),'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source).hexdigest(),'protocol_sha256':hashlib.sha256(protocol).hexdigest(),
        'model_source_sha256':hashlib.sha256(source_model.read_bytes()).hexdigest(),'checkpoint':str(checkpoint),
        'language_seed':123,'action_seeds':args.action_seeds,'precision':args.precision,
        'euler_steps':args.euler_steps,'heun_steps':args.heun_steps,'contexts':len(paths),'limitations':['One map and one recorded route','Reference solvers are not correct-action or safety oracles',
        'Frozen language branch, no independent language uncertainty replication','No learned error estimator or novelty established']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
