"""Primary analysis of the WOD-E2E study (protocols/wod_e2e.md, sections 3-7 and addendum 1), first 280 contexts.

Per segment: mean over the 6 seeds of RFS-proxy (5 s, 10 Hz plan), ADE_1 against the logged 5 s future at 4 Hz, spread (mean pairwise
plan distance between seeds, as paper_numbers.clip_metrics) and flips (decision-class change vs the Heun-32 tau=1 plan of the same seed,
6.4 s plan). tau_eff by spread matching on the Heun-32 curve; residual = sampler - curve metric at tau_eff; segment bootstrap
(1000 draws as registered, seed 20261029 — the protocol registers no seed; the audit of 29/10 found the E4 rater-score verdict seed-dependent, so --boot 10000 is also run and reported as the converged verdict) re-solving tau_eff; TOST at 90% with margins RFS 0.08, ADE_1 0.05 m, flips 2 pt.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.wod_e2e_rfs import rfs,alpamayo_times,at_time,RATER_DT

TAUS=[1.,.9,.8,.7,.6,.5,.4,.3,.2,.1];CURVE=[f'heun32_t{t}' for t in TAUS];SAMPLERS=['euler10_t1.0','euler4_t1.0','euler1_t1.0'];REF='heun32_t1.0'
MARGINS={'rfs':.08,'ade1':.05,'flip':.02};SEED=20261029;BOOT=1000;PT=alpamayo_times(64);FT=RATER_DT*np.arange(1,21)


def ade_log(plan,future):
    p=np.stack([at_time(plan[:,:2],PT,t) for t in FT]);return float(np.linalg.norm(p-future[:,:2],axis=1).mean())


def ade_rater(plan,rater_xy,scores):
    r=rater_xy[int(np.argmax(scores))];t=RATER_DT*np.arange(1,len(r)+1);k=~np.isnan(r).any(1)
    p=np.stack([at_time(plan[:,:2],PT,x) for x in t[k]]);return float(np.linalg.norm(p-r[k],axis=1).mean())


def segment_table(plans,n,seeds,ctx,speed):
    out={};ref_dec=[decisions(plans[f'{n}_s{s}_{REF}'],speed) for s in seeds];key=lambda x:(x['longitudinal'],x['lateral'])
    for c in CURVE+SAMPLERS:
        tr=[plans[f'{n}_s{s}_{c}'] for s in seeds];d=[decisions(t,speed) for t in tr]
        out[c]={'rfs':float(np.mean([rfs(t[:,:2],PT,ctx['rater_xy'],ctx['rater_scores']) for t in tr])),
            'ade1':float(np.mean([ade_log(t,ctx['wod_future_xyz']) for t in tr])),
            'ade_rater':float(np.mean([ade_rater(t,ctx['rater_xy'],ctx['rater_scores']) for t in tr])),
            'spread':float(np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(tr) for b in tr[i+1:]])),
            'flip':float(np.mean([key(a)!=key(b) for a,b in zip(d,ref_dec)]))}
    return out


def matched(T,idx,sampler,metrics=('rfs','ade1','flip')):
    M=lambda c,m:float(T[c][m][idx].mean());sp=np.array([M(c,'spread') for c in CURVE]);o=np.argsort(sp);x=M(sampler,'spread')
    out={'tau_eff':float(np.interp(x,sp[o],np.array(TAUS)[o])),'in_range':bool(sp.min()<=x<=sp.max()),'spread':x}
    for m in metrics:
        at=float(np.interp(x,sp[o],np.array([M(c,m) for c in CURVE])[o]));v=M(sampler,m);out[m]={'sampler':v,'matched':at,'residual':v-at,'raw_effect':v-M(REF,m)}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('artifacts/wod_e2e_val_contexts_v2_20261029'))
    p.add_argument('--probe',default='artifacts/wod_e2e_probe_a15_v1_20261029_p{}');p.add_argument('--out',type=Path,default=Path('artifacts/wod_e2e_analysis_primary_v1_20261029'))
    p.add_argument('--n',type=int,default=280,help='280 = registered primary sample; 479 = all rated segments (secondary)');p.add_argument('--shards',type=int,default=4)
    p.add_argument('--boot',type=int,default=BOOT);p.add_argument('--seed',type=int,default=SEED)
    a=p.parse_args();records={r['name']:r for r in json.loads((a.data/'records.json').read_text())}
    names=[f'context_{i:03d}' for i in range(a.n)];assert all('error' not in records[n] for n in names)
    plans={};seeds=None
    for i in range(a.shards):
        d=Path(a.probe.format(i));m=json.loads((d/'manifest.json').read_text());seeds=seeds or m['seeds'];assert m['seeds']==seeds
        with np.load(d/'plans.npz') as z:plans.update({k:z[k] for k in z.files})
    per={};refl={}
    for n in names:
        c=np.load(a.data/f'{n}.npz');ctx={k:c[k] for k in ('rater_xy','rater_scores','wod_future_xyz','ego_future_xyz')}
        per[n]=segment_table(plans,n,seeds,ctx,records[n]['speed_mps'])
        refl[n]={'log_future':rfs(ctx['wod_future_xyz'][:,:2],FT,ctx['rater_xy'],ctx['rater_scores']),
            'best_rater':float(np.nanmax([rfs(ctx['rater_xy'][k],RATER_DT*np.arange(1,22),ctx['rater_xy'],ctx['rater_scores']) for k in range(len(ctx['rater_scores'])) if ctx['rater_scores'][k]>=0])),  # nanmax: a NaN first item must not drop the segment
            'max_rater_score':float(ctx['rater_scores'].max())}
        refl[n]['ego_future_10hz_check']=rfs(ctx['ego_future_xyz'][:50,:2],PT[:50],ctx['rater_xy'],ctx['rater_scores'])
    ok=[i for i,n in enumerate(names) if not any(np.isnan(per[n][c]['rfs']) for c in CURVE+SAMPLERS)];dropped=[names[i] for i in range(a.n) if i not in ok]
    names_ok=[names[i] for i in ok];segs=[records[n]['segment'] for n in names_ok]
    T={c:{m:np.array([per[n][c][m] for n in names_ok]) for m in ('rfs','ade1','ade_rater','spread','flip')} for c in CURVE+SAMPLERS};N=len(names_ok)
    rng=np.random.default_rng(a.seed);draws=[rng.integers(0,N,N) for _ in range(a.boot)];ALL=np.arange(N)
    res={'n_segments':N,'n_distinct_segments':len(set(segs)),'dropped_nan':dropped,'seeds':seeds,'boot':a.boot,'boot_seed':a.seed,'margins':MARGINS,
        'configs':{c:{m:float(T[c][m].mean()) for m in ('rfs','ade1','ade_rater','spread','flip')} for c in CURVE+SAMPLERS}}
    res['reference_levels']={k:float(np.nanmean([refl[n][k] for n in names_ok])) for k in ('log_future','best_rater','max_rater_score','ego_future_10hz_check')}
    res['reference_levels']['euler10']=res['configs']['euler10_t1.0']['rfs'];res['reference_levels']['log_future_ci95']=np.quantile([np.nanmean([refl[names_ok[i]]['log_future'] for i in d]) for d in draws],[.025,.975]).tolist()
    res['rfs_range_over_configs']=float(max(v['rfs'] for v in res['configs'].values())-min(v['rfs'] for v in res['configs'].values()))
    res['metric_uninformative']=bool(res['rfs_range_over_configs']<.1)
    res['samplers']={}
    for s in SAMPLERS:
        m=matched(T,ALL,s);B=[matched(T,d,s) for d in draws]
        m['tau_eff_ci95']=np.quantile([b['tau_eff'] for b in B],[.025,.975]).tolist();m['boot_in_range']=float(np.mean([b['in_range'] for b in B]))
        for q in MARGINS:
            r=[b[q]['residual'] for b in B];e=[b[q]['raw_effect'] for b in B]
            m[q]['residual_ci90']=np.quantile(r,[.05,.95]).tolist();m[q]['residual_ci95']=np.quantile(r,[.025,.975]).tolist()
            m[q]['equivalent']=bool(m['in_range'] and -MARGINS[q]<m[q]['residual_ci90'][0] and m[q]['residual_ci90'][1]<MARGINS[q])
            m[q]['raw_effect_ci95']=np.quantile(e,[.025,.975]).tolist();m[q]['raw_effect_includes_0']=bool(m[q]['raw_effect_ci95'][0]<=0<=m[q]['raw_effect_ci95'][1])
        res['samplers'][s]=m
    d4=[T['heun32_t0.4']['rfs'][d].mean()-T[REF]['rfs'][d].mean() for d in draws];pt=float(T['heun32_t0.4']['rfs'].mean()-T[REF]['rfs'].mean())
    res['W5_detail']={'rfs_t0.4_minus_t1':pt,'ci95':np.quantile(d4,[.025,.975]).tolist(),'includes_0':bool(np.quantile(d4,.025)<=0<=np.quantile(d4,.975))}
    S=res['samplers'];e10,e4,e1=S['euler10_t1.0'],S['euler4_t1.0'],S['euler1_t1.0']
    res['W']={'W1':bool(e10['rfs']['equivalent'] and e4['rfs']['equivalent']),
        'W2':bool(e10['in_range'] and .75<=e10['tau_eff']<=.92 and e4['in_range'] and .5<=e4['tau_eff']<=.75 and e1['tau_eff']<=.45),
        'W2_parts':{'E10':bool(.75<=e10['tau_eff']<=.92),'E4':bool(.5<=e4['tau_eff']<=.75),'E1':bool(e1['tau_eff']<=.45)},
        'W3':bool(e10['ade1']['equivalent'] and e4['ade1']['equivalent']),'W4_E10':bool(e10['flip']['equivalent']),'W4_E4_E1_informational':[bool(e4['flip']['equivalent']),bool(e1['flip']['equivalent'])],
        'W5':bool(pt>=.05)}
    res['inconclusive_E10_tau_below_grid']=bool(e10['tau_eff']<.1)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1));(a.out/'per_segment.json').write_text(json.dumps({'per_segment':per,'reference':refl},indent=1))
    (a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in res.items() if k not in('samplers',)},indent=1))
    for s,m in S.items():
        print(s,'tau',round(m['tau_eff'],3),[round(x,3) for x in m['tau_eff_ci95']],m['in_range'])
        for q in MARGINS:print('  ',q,{k:(np.round(v,4).tolist() if not isinstance(v,bool) else v) for k,v in m[q].items()})


if __name__=='__main__':main()
