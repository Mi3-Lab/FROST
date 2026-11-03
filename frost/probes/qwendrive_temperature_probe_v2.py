"""Numerical-temperature probe for Qwen-Drive-1.0, v2: reference solver inside the trained time range.

Qwen-Drive's planner predicts the clean trajectory and is trained only for flow times
t <= 0.9; its release sampler floors 1-t at 0.1, so the 10-step sampler ends on the
endpoint predicted at t = 0.9. The v1 probe's 64-step reference queried t > 0.9, which the
network never saw. Here the reference integrates the same ODE (velocity = (endpoint - x) /
(1 - t)) with Heun on [0, 0.9] only and then lands on the endpoint predicted at t = 0.9,
i.e. the release sampler's own final step with negligible discretization error.

Scenes follow docs/data.md of the Qwen-Drive release (as in v1). Planning uses the documented
evaluation mode (reasoning_planning, greedy reasoning), so every sampler of a clip shares
one reasoning trace. Samples use the release's per-sample seeds (noise_seed + k); tau
multiplies the initial noise. With --expert-fp32 the planning expert and the scene cache
run in float32, so spreads are not quantized by bfloat16 outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from frost.probes.qwendrive_temperature_probe import build_scene,TURN_RAD

T_END=.9
TAUS=[1.,.9,.8,.7,.6,.5,.4,.3]
CONFIGS=[('euler',k,1.) for k in [1,2,4,10]]+[('heun',32,t) for t in TAUS]+[('heun',64,1.)]


def plan(model,cache,anchor,inputs,num_samples,method,steps,tau,fp32):
    """Normalized trajectories from one sampler; 'euler' is the release sampler, 'heun' the reference."""
    from qwen_drive.modeling_qwen_drive import normalize_history,denormalize_trajectory
    expert=model.planning_expert;device=anchor.device;scale=model.trajectory_scale(device)
    tile=lambda x:x.repeat_interleave(num_samples,dim=0)
    history=tile(normalize_history(inputs['history'].float(),scale));nav=tile(inputs['nav_command'])
    ego=tile(inputs['ego_status'].float());position=anchor.expand(-1,num_samples)
    if fp32:cache=[(k.float(),v.float()) for k,v in cache]
    noise=tau*model.config.noise_init_std*model._initial_noise(num_samples,model.config.num_future_points,model.config.noise_seed,device)
    if method=='euler':
        out=expert.sample(scene_cache=cache,position_anchor=position,history=history,history_velocity=tile(inputs['history_velocity'].float()),
            history_acceleration=tile(inputs['history_acceleration'].float()),nav_command=nav,ego_status=ego,noise=noise,num_steps=steps,
            min_one_minus_t=model.config.min_one_minus_t)
        return denormalize_trajectory(out,scale).cpu().numpy()
    queries=expert.encode_history(history,nav,tile(inputs['history_velocity'].float()),tile(inputs['history_acceleration'].float()))
    endpoint=lambda x,t:expert.predict_endpoint(x,torch.full((x.shape[0],),t,device=device,dtype=torch.float32),queries,cache,position,nav,ego)
    velocity=lambda x,t:(endpoint(x,t)-x)/(1.-t)
    x=noise.float();h=T_END/steps
    for i in range(steps):
        t=i*h;v=velocity(x,t);x=x+.5*h*(v+velocity(x+h*v,t+h))
    return denormalize_trajectory(endpoint(x,T_END),scale).cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--planner',default='planner-rl');p.add_argument('--range',type=int,nargs=2,default=[100,160])
    p.add_argument('--samples',type=int,default=6);p.add_argument('--mode',default='reasoning_planning',choices=['reasoning_planning','direct_planning'])
    p.add_argument('--expert-fp32',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=args.resume)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    import physical_ai_av
    from qwen_drive import QwenDriveForPlanning
    records=sorted((r for r in json.loads(args.records.read_text()) if 'error' not in r),key=lambda r:r['name'])[args.range[0]:args.range[1]]
    model=QwenDriveForPlanning.from_pretrained(args.model,planner=str(Path(args.model)/args.planner),dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').cuda().eval()
    if args.expert_fp32:
        # flash-attn has no float32 kernel; the expert's own SDPA branch computes the same attention.
        model.planning_expert.float()
        for module in model.planning_expert.modules():
            if getattr(module,'attn_implementation',None)=='flash_attention_2':module.attn_implementation='sdpa'
    reasoning=args.mode=='reasoning_planning';avdi=physical_ai_av.PhysicalAIAVDatasetInterface();seeds=[model.config.noise_seed+k for k in range(args.samples)]
    plans,traces={},{}
    if args.resume and (args.out/'plans.npz').exists():
        with np.load(args.out/'plans.npz') as z:plans={k:z[k] for k in z.files}
        if (args.out/'reasoning.json').exists():traces=json.loads((args.out/'reasoning.json').read_text())
    done={k.split('_s')[0] for k in plans}
    for record in records:
        name=record['name']
        if name in done:continue
        scene,future=build_scene(avdi,record['clip_id'],record['t0_us'])
        inputs={k:v.to(model.device) if torch.is_tensor(v) else v for k,v in model.processor(scene,with_reasoning=reasoning,device='cpu').items()}
        with torch.no_grad():
            if reasoning:cache,anchor,traces[name]=model._prefill_with_reasoning(inputs,model.config.max_reasoning_tokens)
            else:cache,anchor=model._prefill(inputs)
            for method,steps,tau in CONFIGS:
                for i,trajectory in enumerate(plan(model,cache,anchor,inputs,args.samples,method,steps,tau,args.expert_fp32)):
                    plans[f'{name}_s{seeds[i]}_{method}{steps}_t{tau}']=trajectory
        plans[f'{name}_human']=future
        np.savez_compressed(args.out/'plans.npz',**plans);(args.out/'reasoning.json').write_text(json.dumps(traces,indent=1))
        print(json.dumps({'context':name,'nav_command':scene.nav_command,'plans':len(plans)}),flush=True);torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'model':args.model,'planner':args.planner,'mode':args.mode,'expert_fp32':args.expert_fp32,
        'noise_init_std':model.config.noise_init_std,'min_one_minus_t':model.config.min_one_minus_t,'t_end':T_END,'range':args.range,'seeds':seeds,
        'configs':CONFIGS,'reference':'heun32_t1.0','turn_threshold_rad':TURN_RAD,'source_sha256':hashlib.sha256(source).hexdigest(),
        'limitations':['Navigation command derived from the logged future heading (same for all configurations)','Trajectories are (x, y, heading) at 10 Hz over 5 s']},indent=2))


if __name__=='__main__':main()
