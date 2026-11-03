"""Compare integration schedules at matched effective temperature (seed spread).

For each clip we compute seed spread, decision flips against Heun32 (tau=1),
decision entropy and single-sample ADE to the human future, for the explicit
temperature curve Heun32*tau and for every schedule. The residual of a schedule
is its metric minus the temperature curve interpolated at the schedule's spread;
clip bootstrap recomputes both sides.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.core.temperature_report import entropy


def clip_metrics(trajs,refs,human,speed):
    d=[decisions(t,speed) for t in trajs];r=[decisions(t,speed) for t in refs];key=lambda x:(x['longitudinal'],x['lateral'])
    return {'spread':float(np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]])),
        'flip':float(np.mean([key(a)!=key(b) for a,b in zip(d,r)])),'entropy':entropy([key(x) for x in d]),
        'ade_human':float(np.mean([np.linalg.norm(t[:,:2]-human[:,:2],axis=1).mean() for t in trajs]))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--temperature',type=Path,required=True);p.add_argument('--schedules',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    seeds=json.loads((args.temperature/'manifest.json').read_text())['seeds']
    with np.load(args.temperature/'plans.npz') as z:temp={k:z[k] for k in z.files}
    with np.load(args.schedules/'plans.npz') as z:sched={k:z[k] for k in z.files}
    names=sorted({k.split('_s')[0] for k in sched});taus=[1.,.9,.8,.7,.6,.5,.4]
    methods=sorted({k.split('_',3)[-1] for k in sched})
    per=collections.defaultdict(dict)
    for n in names:
        human=np.load(args.data/f'{n}.npz')['ego_future_xyz'];sp=records[n]['speed_mps']
        refs=[temp[f'{n}_s{s}_heun32_t1.0'] for s in seeds]
        for tau in taus:per[f'temp{tau}'][n]=clip_metrics([temp[f'{n}_s{s}_heun32_t{tau}'] for s in seeds],refs,human,sp)
        for m in methods:per[m][n]=clip_metrics([sched[f'{n}_s{s}_{m}'] for s in seeds],refs,human,sp)
    rng=np.random.default_rng(0);metrics=['flip','entropy','ade_human'];out={}
    def summary(sample):
        curve={k:np.array([np.mean([per[f'temp{t}'][n][k] for n in sample]) for t in taus]) for k in ['spread']+metrics}
        o=np.argsort(curve['spread']);res={}
        for m in methods:
            s=np.mean([per[m][n]['spread'] for n in sample])
            res[m]={'spread':s,'tau_eff':float(np.interp(s,curve['spread'][o],np.array(taus)[o])),
                **{k:np.mean([per[m][n][k] for n in sample]) for k in metrics},
                **{f'{k}_residual':np.mean([per[m][n][k] for n in sample])-float(np.interp(s,curve['spread'][o],curve[k][o])) for k in metrics}}
        return res
    point=summary(names);boots=[summary(list(rng.choice(names,len(names)))) for _ in range(300)]
    for m in methods:
        out[m]={k:[float(point[m][k]),float(np.quantile([b[m][k] for b in boots],.025)),float(np.quantile([b[m][k] for b in boots],.975))] for k in point[m]}
    # Paired residual differences against uniform at the same NFE.
    for m in methods:
        method,nfe=m.rsplit('_',1);base=f'uniform_{nfe}'
        if base in methods and m!=base:
            out[m]['flip_residual_minus_uniform']=[float(point[m]['flip_residual']-point[base]['flip_residual']),
                *[float(q) for q in np.quantile([b[m]['flip_residual']-b[base]['flip_residual'] for b in boots],[.025,.975])]]
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'clips':len(names),'seeds':seeds,'results':out},indent=2))
    print(f"{'schedule':32s} tau_eff  flip   flip_resid [CI]            resid-uniform [CI]          H_resid   ADEhum_resid")
    for m in sorted(methods,key=lambda x:(int(x.rsplit('_',1)[1]),x)):
        r=out[m];d=r.get('flip_residual_minus_uniform')
        print(f"{m:32s} {r['tau_eff'][0]:.2f}   {r['flip'][0]:.3f}  {r['flip_residual'][0]:+.3f} [{r['flip_residual'][1]:+.3f},{r['flip_residual'][2]:+.3f}]  "
            +(f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" if d else ' '*26)+f"   {r['entropy_residual'][0]:+.3f}   {r['ade_human_residual'][0]:+.3f}")


if __name__=='__main__':main()
