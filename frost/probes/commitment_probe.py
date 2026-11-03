"""Decision commitment time and windowed susceptibility of a frozen flow driving policy.

Physics analogy (speciation in diffusion models, Biroli et al. 2024): the
generative flow commits to a decision class at some time t_c. We measure (i) the
class of the endpoint prediction x+(1-t)v along a fine reference integration and
(ii) causal susceptibility: freeze the velocity (cache) or kick the state inside
one time window of a uniform Euler20 integration and record decision flips.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from frost.probes.alpamayo_field import load_alpamayo, prepare_field
from frost.core.decisions import decisions

WINDOWS=[(0,2),(2,4),(4,6),(6,8),(8,10)]


def euler(field,initial,steps,freeze=None,kick=None,kick_direction=None,record=False):
    """Uniform Euler; freeze=(a,b) reuses the last fresh velocity for steps a..b-1; kick=(step,eps) perturbs x."""
    x=initial.clone();v=None;evals=0;endpoints=[]
    for i in range(steps):
        t=i/steps
        if kick is not None and i==kick[0]:x=x+kick[1]*x.norm()*kick_direction
        frozen=freeze is not None and freeze[0]<=i<freeze[1] and v is not None
        if not frozen:v=field(x,torch.full((len(x),1,1),t,device=x.device));evals+=1
        if record:endpoints.append((t,(x+(1-t)*v).clone()))
        x=x+v/steps
    return x,evals,endpoints


def heun(field,initial,steps):
    x=initial.clone();endpoints=[]
    for i in range(steps):
        t=i/steps;dt=1/steps;tt=torch.full((len(x),1,1),t,device=x.device)
        v=field(x,tt);endpoints.append((t,(x+(1-t)*v).clone()))
        v2=field(x+dt*v,tt+dt);x=x+.5*dt*(v+v2)
    return x,endpoints


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(9201,9207)));p.add_argument('--kick-eps',type=float,default=.05)
    p.add_argument('--reference-steps',type=int,default=16);p.add_argument('--base-steps',type=int,default=10)
    p.add_argument('--no-kick',action='store_true');p.add_argument('--limit',type=int,default=None)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    contexts={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    model,helper,processor=load_alpamayo();rows=[];outputs={}
    paths=sorted(args.data.glob('context_*.npz'))[:args.limit]
    for path in paths:
        field,dims,hx,hr,official=prepare_field(model,helper,processor,path);speed=contexts[path.stem]['speed_mps']
        def traj(action):
            xyz,_=model.action_space.action_to_traj(action,hx[:,0],hr[:,0]);return xyz.float().cpu().numpy().reshape(-1,3)
        key=lambda d:f"{d['longitudinal']}|{d['lateral']}"
        for seed in args.seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=torch.randn(1,*dims,device='cuda')
            direction=torch.randn(1,*dims,device='cuda',generator=torch.Generator('cuda').manual_seed(seed+777));direction/=direction.norm()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                ref_action,ref_endpoints=heun(field,initial,args.reference_steps)
                ref_plan=traj(ref_action);ref_class=key(decisions(ref_plan,speed))
                endpoint_classes=[(t,key(decisions(traj(e),speed))) for t,e in ref_endpoints]
                committed=next((t for i,(t,_) in enumerate(endpoint_classes) if all(c==ref_class for _,c in endpoint_classes[i:])),1.)
                base_action,_,base_endpoints=euler(field,initial,args.base_steps,record=True);base_plan=traj(base_action);base_class=key(decisions(base_plan,speed))
                cached_action,_,_=euler(field,initial,8,freeze=(2,6));cached_class=key(decisions(traj(cached_action),speed))
                freeze,kick=[],[]
                for a,b in WINDOWS:
                    action,evals,_=euler(field,initial,args.base_steps,freeze=(a,b));plan=traj(action);c=key(decisions(plan,speed))
                    freeze.append({'window':[a/args.base_steps,b/args.base_steps],'evals':evals,'flip_vs_base':c!=base_class,'flip_vs_reference':c!=ref_class,
                        'ade_vs_base':float(np.linalg.norm(plan[:,:2]-base_plan[:,:2],axis=1).mean())})
                    if args.no_kick:continue
                    action,_,_=euler(field,initial,args.base_steps,kick=(a,args.kick_eps),kick_direction=direction);plan=traj(action);c=key(decisions(plan,speed))
                    kick.append({'time':a/args.base_steps,'flip_vs_base':c!=base_class,'ade_vs_base':float(np.linalg.norm(plan[:,:2]-base_plan[:,:2],axis=1).mean())})
            name=f'{path.stem}_s{seed}';outputs[name+'_reference']=ref_plan;outputs[name+'_base']=base_plan
            row={'context':path.stem,'seed':seed,'speed_mps':speed,'reference_class':ref_class,'base_class':base_class,'flashdrive_cached8_class':cached_class,
                'commit_time':committed,'endpoint_classes':endpoint_classes,'freeze':freeze,'kick':kick}
            rows.append(row);print(json.dumps({k:row[k] for k in ['context','seed','reference_class','base_class','flashdrive_cached8_class','commit_time']}|
                {'freeze_flips':[f['flip_vs_base'] for f in freeze],'kick_flips':[k['flip_vs_base'] for k in kick]}),flush=True)
        (args.out/'rows.json').write_text(json.dumps(rows));np.savez_compressed(args.out/'plans.npz',**outputs)
        del field;torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'data':str(args.data),'seeds':args.seeds,'kick_eps':args.kick_eps,
        'reference':f'heun{args.reference_steps}','base':f'euler{args.base_steps}','kick':not args.no_kick,'windows':WINDOWS,'contexts':len(paths),'source_sha256':hashlib.sha256(source).hexdigest(),
        'limitations':['Endpoint-prediction class is a proxy for commitment, not a physical order parameter',
            'Language branch fixed by seed 123','Heun reference has residual discretization error','Decision classes are coarse plan thresholds']},indent=2))


if __name__=='__main__':main()
