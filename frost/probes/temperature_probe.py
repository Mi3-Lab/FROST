"""Explicit versus implicit (numerical) sampling temperature for a frozen flow driving policy.

Implicit: few-step Euler shrinks samples toward the conditional mean. Explicit:
scale the initial noise z by tau and integrate accurately. For each clip and
seed we record plans for both families, so Pareto fronts of single-sample error,
multi-sample coverage and decision entropy can be compared at equal NFE.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from frost.probes.alpamayo_field import load_alpamayo, prepare_field

TAUS=[1.,.9,.8,.7,.6,.5,.4]


def integrate(field,initial,steps,method):
    x=initial.clone();h=1/steps
    for i in range(steps):
        t=torch.full((len(x),1,1),i*h,device=x.device);v=field(x,t)
        x=x+h*v if method=='euler' else x+.5*h*(v+field(x+h*v,t+h))
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--range',type=int,nargs=2,default=[100,200]);p.add_argument('--seeds',type=int,nargs='+',default=list(range(9201,9207)))
    p.add_argument('--configs',nargs='+',default=None,help='method:steps:tau triples replacing the default grid (same seeds, same initial noise)')
    p.add_argument('--resume',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=args.resume);torch.set_num_threads(2)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    names=sorted(r['name'] for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r)[args.range[0]:args.range[1]]
    model,helper,processor=load_alpamayo();plans={};configs=([('euler',k,1.) for k in [1,2,4,6,10]]+[('heun',k,tau) for k in [3,5] for tau in TAUS]
        +[('euler',10,tau) for tau in TAUS[1:]]+[('heun',32,tau) for tau in TAUS])
    if args.configs:configs=[(m,int(k),float(t)) for m,k,t in (c.split(':') for c in args.configs)]
    if args.resume and (args.out/'plans.npz').exists():
        with np.load(args.out/'plans.npz') as z:plans={k:z[k] for k in z.files}
    done={k.split('_s')[0] for k in plans}
    for name in names:
        if name in done:continue
        field,dims,hx,hr,_=prepare_field(model,helper,processor,args.data/f'{name}.npz')
        for seed in args.seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=torch.randn(1,*dims,device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for method,steps,tau in configs:
                    xyz,_=model.action_space.action_to_traj(integrate(field,tau*initial,steps,method),hx[:,0],hr[:,0])
                    plans[f'{name}_s{seed}_{method}{steps}_t{tau}']=xyz.float().cpu().numpy().reshape(-1,3)
        np.savez_compressed(args.out/'plans.npz',**plans);print(json.dumps({'context':name,'plans':len(plans)}),flush=True)
        del field;torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'data':str(args.data),'range':args.range,'seeds':args.seeds,
        'configs':configs,'nfe':{f'{m}{k}':k*(2 if m=='heun' else 1) for m,k,_ in configs},'source_sha256':hashlib.sha256(source).hexdigest()},indent=2))


if __name__=='__main__':main()
