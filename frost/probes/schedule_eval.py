"""Equal-NFE schedule comparison on frozen Alpamayo1.5 with calibration/evaluation clip split.

calibrate: on calibration clips, estimate (a) decision-information density from the
  commitment probe (agreement of endpoint-prediction class with the final class),
  (b) field-turn density 1-cos(v_i,v_i+1) from stored Euler10 velocities (ProbeFlow-like),
  (c) rescaled entropic-time density, which in x_t=(1-t)z+t x1 reduces to the
  flow-matching loss E|v(x_t,t)-(x1-z)|^2 with x1 = action of the logged human future.
evaluate: on held-out clips, integrate Euler on each schedule's grid and compare decision
  classes with the Heun160 reference (same noise) and with the human decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from frost.probes.alpamayo_field import load_alpamayo, prepare_field
from frost.core.schedules import grid_from_density
from frost.core.decisions import decisions

GRID=np.linspace(0,1,17)[:-1]


def density_from_agreement(times,agreement):
    gain=np.maximum(np.diff(np.r_[agreement,1.]),0);return gain/max(gain.sum(),1e-12)


def euler_grid(field,initial,grid):
    x=initial.clone()
    for a,b in zip(grid[:-1],grid[1:]):x=x+float(b-a)*field(x,torch.full((len(x),1,1),float(a),device=x.device))
    return x


def cached8(field,initial):
    x=initial.clone();v=None
    for i in range(8):
        if not 2<=i<6:v=field(x,torch.full((len(x),1,1),i/8,device=x.device))
        x=x+v/8
    return x


def calibrate(args,model,helper,processor,contexts,names):
    rows=[r for r in json.loads((args.commitment/'rows.json').read_text()) if r['context'] in names]
    agreement=np.array([np.mean([r['endpoint_classes'][i][1]==r['reference_class'] for r in rows]) for i in range(len(GRID))])
    with np.load(args.solver_predictions/'predictions.npz') as z:
        turns=[]
        for name in names:
            for seed in [9201,9202]:
                v=z[f'{name}_s{seed}_euler10_velocities'].reshape(10,-1)
                turns.append(1-np.sum(v[1:]*v[:-1],1)/(np.linalg.norm(v[1:],axis=1)*np.linalg.norm(v[:-1],axis=1)))
    turn=np.r_[np.mean(turns,0),0.];turn_times=np.linspace(0,1,11)[:-1]
    loss_times=np.linspace(0,1,21)[:-1]+.025;losses=[]
    for name in names[:args.loss_contexts]:
        field,dims,hx,hr,_=prepare_field(model,helper,processor,args.data/f'{name}.npz')
        with np.load(args.data/f'{name}.npz') as z:
            fx=torch.from_numpy(z['ego_future_xyz'])[None].cuda().float();fr=torch.from_numpy(z['ego_future_rot'])[None].cuda().float()
        with torch.inference_mode():
            x1=model.action_space.traj_to_action(hx[:,0].float(),hr[:,0].float(),fx,fr).reshape(1,*dims).float()
            gen=torch.Generator('cuda').manual_seed(4242);per_t=[]
            for t in loss_times:
                values=[]
                for _ in range(args.loss_noise):
                    z0=torch.randn(1,*dims,device='cuda',generator=gen);xt=(1-t)*z0+t*x1
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=field(xt,torch.full((1,1,1),float(t),device='cuda'))
                    values.append(float(((pred.float()-(x1-z0))**2).mean()))
                per_t.append(np.mean(values))
        losses.append(per_t);print(json.dumps({'calibration_loss_context':name,'loss':[round(v,4) for v in per_t]}),flush=True)
        del field;torch.cuda.empty_cache()
    return {'grid':GRID.tolist(),'decision_agreement':agreement.tolist(),'turn_times':turn_times.tolist(),'turn':turn[:-1].tolist(),
        'loss_times':loss_times.tolist(),'fm_loss':np.mean(losses,0).tolist(),'calibration_contexts':names,'commitment_rows':len(rows)}


def schedules(curves,nfe):
    t=np.array(curves['grid']);dec=density_from_agreement(t,np.array(curves['decision_agreement']))
    dense=np.linspace(0,1,201)[:-1]
    def on_dense(times,values):
        idx=np.clip(np.searchsorted(times,dense,side='right')-1,0,len(values)-1);return np.asarray(values)[idx]
    d=on_dense(t,dec/np.diff(np.r_[t,1.]));turn=on_dense(np.array(curves['turn_times']),curves['turn']);loss=np.interp(dense,curves['loss_times'],curves['fm_loss'])
    return {'uniform':np.linspace(0,1,nfe+1),'decision_info':grid_from_density(dense,d+1e-3*d.max(),nfe),
        'decision_info_plus_uniform':grid_from_density(dense,d/d.mean()+1,nfe),'field_turn':grid_from_density(dense,turn+1e-6,nfe),
        'rescaled_entropic':grid_from_density(dense,loss+1e-9,nfe)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['calibrate','evaluate'],required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--commitment',type=Path);p.add_argument('--solver-predictions',type=Path,required=True)
    p.add_argument('--curves',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--calibration',type=int,nargs=2,default=[0,100]);p.add_argument('--evaluation',type=int,nargs=2,default=[100,200])
    p.add_argument('--loss-contexts',type=int,default=40);p.add_argument('--seeds',type=int,nargs='+',default=[9201,9202])
    p.add_argument('--inline-reference',type=int,default=None,help='Heun steps for an inline reference instead of stored heun160')
    p.add_argument('--save-plans',action='store_true');p.add_argument('--loss-noise',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(2)
    contexts={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    model,helper,processor=load_alpamayo()
    if args.phase=='calibrate':
        names=[f'context_{i:03d}' for i in range(*args.calibration) if f'context_{i:03d}' in contexts]
        curves=calibrate(args,model,helper,processor,contexts,names);curves['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.curves.write_text(json.dumps(curves,indent=2));return
    curves=json.loads(args.curves.read_text());args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    grids={nfe:schedules(curves,nfe) for nfe in [4,6,8]}
    (args.out/'grids.json').write_text(json.dumps({n:{k:v.tolist() for k,v in g.items()} for n,g in grids.items()},indent=2))
    refs={}
    if args.inline_reference is None:
        with np.load(args.solver_predictions/'predictions.npz') as z:refs={k:z[k] for k in z.files if k.endswith('_heun160') or k.endswith('_euler8')}
    rows=[];saved={};key=lambda d:f"{d['longitudinal']}|{d['lateral']}"
    names=sorted(contexts)[args.evaluation[0]:args.evaluation[1]] if args.inline_reference else [f'context_{i:03d}' for i in range(*args.evaluation)]
    for name in names:
        if name not in contexts:continue
        field,dims,hx,hr,_=prepare_field(model,helper,processor,args.data/f'{name}.npz');speed=contexts[name]['speed_mps']
        with np.load(args.data/f'{name}.npz') as z:human=z['ego_future_xyz']
        human_class=key(decisions(human,speed))
        def traj(action):
            xyz,_=model.action_space.action_to_traj(action,hx[:,0],hr[:,0]);return xyz.float().cpu().numpy().reshape(-1,3)
        for seed in args.seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=torch.randn(1,*dims,device='cuda')
            if args.inline_reference:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    x=initial.clone();h=1/args.inline_reference
                    for j in range(args.inline_reference):
                        tt=torch.full((1,1,1),j*h,device='cuda');v=field(x,tt);x=x+.5*h*(v+field(x+h*v,tt+h))
                    ref=traj(x)
            else:ref=refs[f'{name}_s{seed}_heun160']
            ref_class=key(decisions(ref,speed))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                plans={f'{method}_{nfe}':traj(euler_grid(field,initial,grid)) for nfe,g in grids.items() for method,grid in g.items()}
                plans['flashdrive_cached8_4']=traj(cached8(field,initial))
            replay=float(np.abs(plans['uniform_8']-refs[f'{name}_s{seed}_euler8']).max()) if refs else 0.
            if replay>.05:raise RuntimeError(f'uniform_8 does not replay native euler8: {replay}')
            if args.save_plans:saved.update({f'{name}_s{seed}_{m}':v for m,v in plans.items()})
            for method,plan in plans.items():
                c=key(decisions(plan,speed))
                rows.append({'context':name,'seed':seed,'method':method.rsplit('_',1)[0],'nfe':int(method.rsplit('_',1)[1]),'class':c,
                    'flip_vs_reference':c!=ref_class,'agree_human':c==human_class,'reference_agree_human':ref_class==human_class,
                    'ade_reference':float(np.linalg.norm(plan[:,:2]-ref[:,:2],axis=1).mean()),'ade_human':float(np.linalg.norm(plan[:,:2]-human[:,:2],axis=1).mean())})
        (args.out/'rows.json').write_text(json.dumps(rows))
        if args.save_plans:np.savez_compressed(args.out/'plans.npz',**saved)
        print(json.dumps({'context':name,'flips':{m:sum(r['flip_vs_reference'] for r in rows if r['context']==name and f"{r['method']}_{r['nfe']}"==m) for m in plans}}),flush=True)
        del field;torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'curves':str(args.curves),'curves_sha256':hashlib.sha256(args.curves.read_bytes()).hexdigest(),
        'evaluation':args.evaluation,'seeds':args.seeds,'data':str(args.data),
        'reference':f'inline heun{args.inline_reference}' if args.inline_reference else 'heun160 from native_solver_physicalai_v1 with identical seeds'},indent=2))


if __name__=='__main__':main()
