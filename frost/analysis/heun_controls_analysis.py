"""Heun-3/5 controls on held-out clips (protocols/frost_main.md, addendum 15).

Heun-k flips are against the Heun-32 tau=1 reference of the same run; spread and ADE_1 of Heun-k come from the control run.
tau_eff and the curve value at tau_eff come from the main run's temperature curve (clip means, flips against the main run's
reference). Clip bootstrap (2000 draws, seed 20261026) re-solving tau_eff; 90% intervals decide equivalence, 95% reported.
Also reports the max |difference| between the two runs' Heun-32 tau=1 plans.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.paper_numbers import TAUS,CURVE,REFERENCE,MARGINS,spread

SAMPLERS=['heun3_t1.0','heun5_t1.0']


def metrics(plans,n,cfg,seeds,speed,human,ref_dec):
    T=[plans[f'{n}_s{s}_{cfg}'] for s in seeds];key=lambda x:(x['longitudinal'],x['lateral'])
    return {'spread':spread(T),'ade1':float(np.mean([np.linalg.norm(t[:,:2]-human[:,:2],axis=1).mean() for t in T])),
        'flip':float(np.mean([key(decisions(t,speed))!=key(r) for t,r in zip(T,ref_dec)]))}


def table(path,cfgs,names,data,records):
    seeds=json.loads((path/'manifest.json').read_text())['seeds'];out={c:{} for c in cfgs}
    with np.load(path/'plans.npz') as z:plans={k:z[k] for k in z.files}
    for n in names:
        human=np.load(data/f'{n}.npz')['ego_future_xyz'];v=records[n]['speed_mps']
        ref=[decisions(plans[f'{n}_s{s}_{REFERENCE}'],v) for s in seeds]
        for c in cfgs:out[c][n]=metrics(plans,n,c,seeds,v,human,ref)
    return out,plans,seeds


def matched(main,ctrl,sample,s):
    M=lambda t,c,m:float(np.mean([t[c][n][m] for n in sample]))
    sp=np.array([M(main,c,'spread') for c in CURVE]);o=np.argsort(sp);x=M(ctrl,s,'spread')
    out={'tau_eff':float(np.interp(x,sp[o],np.array(TAUS)[o])),'in_range':bool(sp.min()<=x<=sp.max())}
    for m in ['ade1','flip']:
        at=float(np.interp(x,sp[o],np.array([M(main,c,m) for c in CURVE])[o]));v=M(ctrl,s,m)
        out[m]={'sampler':v,'matched':at,'residual':v-at,'effect':v-M(ctrl,REFERENCE,m)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--main',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--boot',type=int,default=2000);a=p.parse_args()
    records={r['name']:r for r in json.loads((a.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*a.range)]
    main_t,main_p,seeds=table(a.main,CURVE,names,a.data,records);ctrl_t,ctrl_p,_=table(a.controls,SAMPLERS+[REFERENCE],names,a.data,records)
    refdiff=float(max(np.abs(main_p[f'{n}_s{s}_{REFERENCE}']-ctrl_p[f'{n}_s{s}_{REFERENCE}']).max() for n in names for s in seeds))
    rng=np.random.default_rng(20261026);draws=[rng.choice(names,len(names)) for _ in range(a.boot)]
    res={'clips':len(names),'margins':MARGINS,'reference_max_abs_diff_m':refdiff,'samplers':{}}
    for s in SAMPLERS:
        m=matched(main_t,ctrl_t,names,s);B=[matched(main_t,ctrl_t,d,s) for d in draws]
        m['tau_eff_ci95']=np.quantile([b['tau_eff'] for b in B],[.025,.975]).tolist()
        for k in ['ade1','flip']:
            r=[b[k]['residual'] for b in B];m[k]['residual_ci95']=np.quantile(r,[.025,.975]).tolist();m[k]['residual_ci90']=np.quantile(r,[.05,.95]).tolist()
            m[k]['equivalent']=bool(m['in_range'] and -MARGINS[k]<m[k]['residual_ci90'][0] and m[k]['residual_ci90'][1]<MARGINS[k])
        m['prediction_15']=bool(m['tau_eff']>=.85 and m['flip']['residual']>.02)
        res['samplers'][s]=m
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1))
    (a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('reference max |diff| (m):',refdiff)
    for s,m in res['samplers'].items():
        print(f"{s}: tau {m['tau_eff']:.3f} {[round(x,3) for x in m['tau_eff_ci95']]} | flip {100*m['flip']['sampler']:.1f} vs {100*m['flip']['matched']:.1f} resid {100*m['flip']['residual']:+.1f} {[round(100*x,1) for x in m['flip']['residual_ci95']]} | ADE resid {m['ade1']['residual']:+.3f} {[round(x,3) for x in m['ade1']['residual_ci95']]} | pred15 {m['prediction_15']}")


if __name__=='__main__':main()
