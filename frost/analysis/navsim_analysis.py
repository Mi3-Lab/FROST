"""Pre-registered NAVSIM second-family analysis (protocols/navsim.md, addenda 1 and 1b).

Per scene and config, means over the six seeds: seed spread (mean pairwise L2 over the 8 waypoints), ADE_1 against
the human future, decision flips against the reference at tau=1 under the same seed (trajectory_decision_gate.decisions,
current speed from the navtest metric cache), and PDMS from navsim_pdm_score_plans. tau_eff interpolates the scene-mean
spread on the reference curve; residual = sampler - curve at tau_eff, effect = sampler - reference at tau=1. 2000-draw
scene bootstrap (seed 20261028) re-solving tau_eff; 90% intervals decide equivalence (TOST), 95% are reported.
Addendum 2: flips use decisions() on the plan resampled to 0.1 s (dense); the registered v1 flips (raw 8-point plan,
longitudinal class degenerate) are kept as flip_registered; tau_eff per axis and against the 100-step reference (linear in tau).
Family recogdrive_lead: addendum-2 leading sweep against the 100-step curve; N1' and N2' (E5/E10 leading).
Run with the diffusiondrive venv (pandas/pyarrow, navsim on PYTHONPATH for the metric-cache pickles).
"""
import argparse
import glob
import json
import lzma
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from frost.core.decisions import decisions

MARGINS={'ade1':0.05,'flip':0.02,'pdms':0.01}  # PDMS margin 1.0 point on the 0-100 scale
FAMILY={'recogdrive':dict(curve='ddim50_trail_eta0_t{}',taus=[1.,.9,.8,.7,.6,.5,.4,.3],
    samplers=['ddim1_trail_eta0_t1.0','ddim2_trail_eta0_t1.0','ddim5_trail_eta0_t1.0','ddim10_trail_eta0_t1.0','ddim5_lead_eta0_t1.0','release5_lead_eta1_t1.0'],
    check='ddim100_trail_eta0_t1.0'),
    'recogdrive_lead':dict(curve='ddim100_trail_eta0_t{}',taus=[1.,.9,.8,.7,.6,.5,.4,.3],  # addendum 2 sweep
        samplers=[f'ddim{k}_lead_eta0_t1.0' for k in (2,4,5,10,20)]+['release5_lead_eta1_t1.0'],check='ddim50_lead_eta0_t1.0'),
    'diffusiondrive':dict(curve='ddim9_int_eta0_t{}',taus=[1.,.9,.8,.7,.6,.5,.4,.3],samplers=['release2_lead_eta0_t1.0'],check='ddim5_int_eta0_t1.0')}


def speeds(cache):
    out={}
    for f in glob.glob(str(cache/'*'/'*'/'*'/'metric_cache.pkl*')):
        try:m=pickle.load(lzma.open(f))
        except lzma.LZMAError:m=pickle.load(open(f,'rb'))
        out[Path(f).parent.name]=float(m.ego_state.dynamic_car_state.speed)
    return out


def dense(t):
    """Addendum 2: 8 waypoints at 0.5 s -> 40 at 0.1 s (origin prepended), the grid decisions() assumes."""
    t=np.asarray(t,float);T=np.arange(0,len(t)+1)*.5;xy=np.vstack([[0,0],t[:,:2]]);g=np.arange(1,int(round(T[-1]/.1))+1)*.1
    return np.c_[np.interp(g,T,xy[:,0]),np.interp(g,T,xy[:,1]),np.zeros(len(g))]


def per_scene(probe,speed):
    z=np.load(probe/'plans.npz');plans=defaultdict(lambda:defaultdict(dict));human={}
    for k in z.files:
        tok,rest=k.split('_',1)
        if rest=='human':human[tok]=z[k][:,:2];continue
        seed,cfg=rest.split('_',1);plans[tok][cfg][seed]=z[k]
    pd_=pd.read_parquet(probe/'pdm_scores.parquet');pd_=pd_[pd_.seed>=0].groupby(['token','config']).pdms.mean()
    key=lambda t,v:(lambda d:(d['longitudinal'],d['lateral']))(decisions(t,v));table={}
    for tok,cfgs in plans.items():
        v=speed[tok];ref=cfgs[REF]
        refdec={s:key(t,v) for s,t in ref.items()};refd={s:key(dense(t),v) for s,t in ref.items()};table[tok]={}
        for cfg,tr in cfgs.items():
            T=list(tr.values());seeds=list(tr)
            pair=lambda f:np.mean([f(a,b) for i,a in enumerate(T) for b in T[i+1:]])
            spread=pair(lambda a,b:np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean())
            table[tok][cfg]=dict(spread=spread,ade1=np.mean([np.linalg.norm(t[:,:2]-human[tok],axis=1).mean() for t in T]),
                flip_registered=np.mean([key(tr[s],v)!=refdec[s] for s in seeds]),flip=np.mean([key(dense(tr[s]),v)!=refd[s] for s in seeds]),
                spread_x=pair(lambda a,b:np.abs(a[:,0]-b[:,0]).mean()),spread_y=pair(lambda a,b:np.abs(a[:,1]-b[:,1]).mean()),pdms=float(pd_.get((tok,cfg),np.nan)))
    return table


def matched(table,toks,fam,s):
    M=lambda cfg,m:np.nanmean([table[t][cfg][m] for t in toks])
    taus=np.array(fam['taus'][::-1]);curve={m:np.array([M(fam['curve'].format(t),m) for t in taus]) for m in ['spread','spread_x','spread_y','ade1','flip','flip_registered','pdms']}
    sp=M(s,'spread');out={'spread':sp,'in_range':bool(curve['spread'][0]<=sp<=curve['spread'][-1])}
    tau=float(np.interp(sp,curve['spread'],taus)) if out['in_range'] else np.nan;out['tau_eff']=tau
    out['tau_eff_censored']=tau if out['in_range'] else (np.inf if sp>curve['spread'][-1] else -np.inf)  # out of range kept as >max / <min tau
    for ax in ['spread_x','spread_y']:out['tau_eff_'+ax[-1]]=float(np.interp(M(s,ax),curve[ax],taus,left=np.nan,right=np.nan))
    if fam['check'] in table[toks[0]]:out['tau_eff_ref100_linear']=sp/M(fam['check'],'spread')  # addendum 2: spread linear in tau
    for m in ['ade1','flip','flip_registered','pdms']:
        v=M(s,m);at=float(np.interp(tau,taus,curve[m])) if out['in_range'] else np.nan
        out[m]={'sampler':v,'reference':float(curve[m][-1]),'matched':at,'effect':v-float(curve[m][-1]),'residual':v-at,'temp_effect':at-float(curve[m][-1])}
    return out


def main():
    global REF
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--family',choices=list(FAMILY),required=True)
    p.add_argument('--cache',type=Path,default=Path('data/navsim_exp/metric_cache'));p.add_argument('--out',type=Path,required=True);p.add_argument('--boot',type=int,default=2000)
    a=p.parse_args();fam=FAMILY[a.family];REF=fam['curve'].format(1.0)
    table=per_scene(a.probe,speeds(a.cache));toks=sorted(table);rng=np.random.default_rng(20261028)
    draws=[rng.choice(toks,len(toks)) for _ in range(a.boot)];res={'scenes':len(toks),'margins':MARGINS,'reference':REF,'samplers':{}}
    conv={m:float(np.max(np.abs([table[t][fam['check']][m]-table[t][REF][m] for t in toks]))) for m in ['ade1','pdms']}
    res['convergence_check']={'config':fam['check'],'max_abs_scene_diff':conv,
        'mean_diff':{m:float(np.mean([table[t][fam['check']][m]-table[t][REF][m] for t in toks])) for m in ['ade1','pdms','spread']}}
    for s in fam['samplers']+[fam['check']]:
        m=matched(table,toks,fam,s);B=[matched(table,d,fam,s) for d in draws]
        m['tau_eff_ci95']=np.nanquantile([b['tau_eff'] for b in B] if any(b['in_range'] for b in B) else [np.nan],[.025,.975]).tolist();m['boot_in_range']=float(np.mean([b['in_range'] for b in B]))
        m['tau_eff_ci95_censored']=np.quantile([b['tau_eff_censored'] for b in B],[.025,.975]).tolist()  # audit 29/10: the in-range-only CI hid out-of-range resamples
        for k in ['ade1','flip','flip_registered','pdms']:
            for f in ['residual','effect','temp_effect']:
                x=[b[k][f] for b in B if b['in_range'] or f=='effect'] or [np.nan];m[k][f+'_ci95']=np.nanquantile(x,[.025,.975]).tolist();m[k][f+'_ci90']=np.nanquantile(x,[.05,.95]).tolist()
            lo,hi=m[k]['residual_ci90'];m[k]['equivalent']=bool(m['in_range'] and -MARGINS[k.split('_')[0]]<lo and hi<MARGINS[k.split('_')[0]])
        res['samplers'][s]=m
    t=lambda s:res['samplers'][s]['tau_eff']
    if a.family=='recogdrive':
        e2,e5,e10='ddim2_trail_eta0_t1.0','ddim5_trail_eta0_t1.0','ddim10_trail_eta0_t1.0'
        res['N1']=bool(t(e2)<t(e5)<t(e10)<=1 and .6<=t(e5)<=.95)
        res['N2']={s:all(res['samplers'][s][k]['equivalent'] for k in MARGINS) for s in [e5,e10]}
        res['N3']=bool(np.sign(res['samplers'][e5]['pdms']['temp_effect'])==np.sign(res['samplers'][e5]['pdms']['effect']))
    if a.family=='recogdrive_lead':
        L=[f'ddim{k}_lead_eta0_t1.0' for k in (2,4,5,10,20,50)];tl=[t(s) for s in L]
        res['N1_prime']=bool(all(res['samplers'][s]['in_range'] for s in L[1:]) and all(np.diff(tl[1:])>0) and not tl[0]>tl[1])
        res['N2_prime']={s:{k:res['samplers'][s][k]['equivalent'] for k in ['pdms','ade1','flip']} for s in L[2:4]}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1,default=float))
    (a.out/'per_scene.json').write_text(json.dumps(table,default=float));(a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('scenes',len(toks),'convergence',res['convergence_check'])
    for s,m in res['samplers'].items():
        f=lambda k,x=1:f"{x*m[k]['sampler']:.3f} eff {x*m[k]['effect']:+.3f} resid {x*m[k]['residual']:+.3f} [{x*m[k]['residual_ci90'][0]:+.3f},{x*m[k]['residual_ci90'][1]:+.3f}] eq={m[k]['equivalent']}"
        print(f"{s:26s} tau_x/y {m['tau_eff_x']:.2f}/{m['tau_eff_y']:.2f} ref100 {m.get('tau_eff_ref100_linear',np.nan):.2f} | flip_reg {100*m['flip_registered']['sampler']:.1f} resid {100*m['flip_registered']['residual']:+.1f}")
        print(f"{s:26s} tau {m['tau_eff']:.3f} {[round(x,3) for x in m['tau_eff_ci95']]} | PDMS {f('pdms',100)} | ADE {f('ade1')} | flip {f('flip',100)}")
    for n in ['N1','N2','N3','N1_prime','N2_prime']:
        if n in res:print(n,res[n])


if __name__=='__main__':main()
