"""Addendum 18 (pre-registered 29/10/2026 14:43 PDT): reference convergence and tau-grid resolution.

Alpamayo-1.5, held-out clips context_000-099, seeds 9201-9206, same initial noise. Curves:
  heun32  : the principal Heun-32 grid tau in {1,.9,.8,.7,.6,.5,.4} (step 0.1; the paper's curve)
  heun64  : Heun-64  at tau in {1,.9,.85,.8,.75,.7}
  heun128 : Heun-128 at tau in {1,.9,.85,.8,.75,.7}
  heun32_dense : principal grid + Heun-32 at {.875,.85,.825,.775,.75} (step 0.025 near tau_eff)
Flips are always counted against the tau=1 reference of the SAME solver as the curve (same seed), for the
curve and for the sampler, as the addendum says. Per-clip metrics come from frost.analysis.paper_numbers.clip_metrics;
tau_eff / residuals use the same spread matching and linear interpolation as frost.analysis.paper_numbers.matched,
parametrized by curve; TOST at alpha=.05 from the 90% bootstrap interval of the residual, margins ADE1 0.05 m,
flips 2 pt (paper_numbers.MARGINS).

Analysis (registered):
 1. Heun-32 minus Heun-128 in spread, ADE1, flips at each tau of {1,.9,.85,.8,.75,.7}, clip CI.
 2. tau_eff and residuals (ADE1, flips) of E10 and E4 against each curve.
 3. Same residuals with heun32_dense vs the principal step-0.1 grid.
 C1 |Heun32 - Heun128| < 0.01 m ADE1 and < 0.5 pt flips at every tau (judged on the point estimate; CI reported).
 C2 tau_eff(E10) differs by < 0.02 among heun32, heun64, heun128 (max pairwise point difference).
 C3 E10 equivalence verdicts (ADE1 +-0.05 m, flips +-2 pt) identical across the three curves and the two grids.
Clip bootstrap: 2000 resamples, seed 20261030, the same resamples for every curve (joint), re-solving tau_eff.
Sensitivity rows labelled 'exploratory' (heun32 restricted to the 64/128 grid; heun64/128 restricted to the
0.1 grid; Heun-64 minus Heun-128) are not part of the registered criteria.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.paper_numbers import clip_metrics, MARGINS

GRID=[1.,.9,.85,.8,.75,.7];PRINCIPAL=[1.,.9,.8,.7,.6,.5,.4];DENSE=[.875,.85,.825,.775,.75]
CURVES={'heun32':('heun32',PRINCIPAL),'heun64':('heun64',GRID),'heun128':('heun128',GRID),
    'heun32_dense':('heun32',sorted(set(PRINCIPAL+DENSE),reverse=True)),
    # exploratory sensitivity: same tau grid for every solver
    'heun32_on_a18_grid':('heun32',GRID),'heun64_step01':('heun64',[1.,.9,.8,.7]),'heun128_step01':('heun128',[1.,.9,.8,.7])}
REGISTERED=['heun32','heun64','heun128','heun32_dense'];SAMPLERS=['euler10_t1.0','euler4_t1.0'];METRICS=['spread','ade1','flip']
cfg=lambda solver,t:f'{solver}_t{t}'


def load(args):
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*args.range)];plans={}
    for probe in (args.principal,args.a18):
        with np.load(probe/'plans.npz') as z:plans.update({k:z[k] for k in z.files if k.split('_s')[0] in names})
    seeds=json.loads((args.a18/'manifest.json').read_text())['seeds']
    assert seeds==json.loads((args.principal/'manifest.json').read_text())['seeds']
    configs=sorted({k.split('_',3)[-1] for k in plans if '_s92' in k})
    tables={}  # tables[solver][config] -> {metric: array over clips}, flips against solver's own tau=1
    for solver in ('heun32','heun64','heun128'):
        rows={c:{m:[] for m in METRICS} for c in configs}
        for n in names:
            human=np.load(args.data/f'{n}.npz')['ego_future_xyz'];speed=records[n]['speed_mps']
            ref_dec=[decisions(plans[f'{n}_s{s}_{solver}_t1.0'],speed) for s in seeds]
            for c in configs:
                r=clip_metrics(plans,n,c,seeds,speed,human,ref_dec)
                for m in METRICS:rows[c][m].append(r[m])
        tables[solver]={c:{m:np.array(v) for m,v in d.items()} for c,d in rows.items()}
    return names,seeds,tables


def matched(table,idx,sampler,curve,taus):
    """paper_numbers.matched on a parametrized curve, for a matrix of clip resamples idx (B, N)."""
    mean=lambda c,m:table[c][m][idx].mean(1)
    spreads=np.stack([mean(c,'spread') for c in curve],1);s=mean(sampler,'spread');taus=np.array(taus)
    out={'tau_eff':np.empty(len(idx)),'in_range':(spreads.min(1)<=s)&(s<=spreads.max(1))}
    curves={m:np.stack([mean(c,m) for c in curve],1) for m in ('ade1','flip')}
    for m in curves:out[m]={'sampler':mean(sampler,m),'matched':np.empty(len(idx)),'reference':mean(curve[0],m)}
    for b in range(len(idx)):
        o=np.argsort(spreads[b]);out['tau_eff'][b]=np.interp(s[b],spreads[b][o],taus[o])
        for m in curves:out[m]['matched'][b]=np.interp(s[b],spreads[b][o],curves[m][b][o])
    for m in curves:out[m]['residual']=out[m]['sampler']-out[m]['matched'];out[m]['effect']=out[m]['sampler']-out[m]['reference']
    return out


ci=lambda x:[float(x[0]),*np.quantile(x[1:],[.025,.975]).tolist()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--principal',type=Path,default=Path('artifacts/temperature_probe_heldout_v3_20261027'))
    p.add_argument('--a18',type=Path,default=Path('artifacts/convergence_a18_a15_heldout_v3_20261029'))
    p.add_argument('--range',type=int,nargs=2,default=[0,100]);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261030);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    names,seeds,tables=load(args);rng=np.random.default_rng(args.seed)
    idx=np.vstack([np.arange(len(names))]+[rng.choice(len(names),len(names)) for _ in range(args.boot)])
    result={'addendum':18,'clips':len(names),'range':args.range,'seeds':seeds,'boot':args.boot,'boot_seed':args.seed,'margins':MARGINS,
        'curves':{k:{'solver':s,'taus':t,'registered':k in REGISTERED} for k,(s,t) in CURVES.items()}}
    # 1. discretization error of the reference: Heun-32 minus Heun-128 (registered); Heun-64 minus Heun-128 (exploratory)
    disc={}
    for a in ('heun32','heun64'):
        rows={}
        for t in GRID:
            row={}
            for m in METRICS:
                d=tables[a][cfg(a,t)][m]-tables['heun128'][cfg('heun128',t)][m];row[m]=ci(d[idx].mean(1))
                row[m+'_abs_per_clip_mean']=float(np.abs(d).mean())
            rows[str(t)]=row
        disc[f'{a}_minus_heun128']=rows
    disc['heun64_minus_heun128']['exploratory']=True;result['discretization']=disc
    worst={m:max(abs(r[m][0]) for t,r in disc['heun32_minus_heun128'].items() if t!='exploratory') for m in ('ade1','flip')}
    worst_ci={m:max(max(abs(r[m][1]),abs(r[m][2])) for t,r in disc['heun32_minus_heun128'].items() if t!='exploratory') for m in ('ade1','flip')}
    result['C1']={'max_abs_ade1_m':worst['ade1'],'max_abs_flip_pt':100*worst['flip'],'max_abs_ci_bound_ade1_m':worst_ci['ade1'],
        'max_abs_ci_bound_flip_pt':100*worst_ci['flip'],'holds':bool(worst['ade1']<.01 and 100*worst['flip']<.5),
        'holds_on_ci_bounds':bool(worst_ci['ade1']<.01 and 100*worst_ci['flip']<.5)}
    # 2./3. tau_eff and residuals of E10 and E4 against each curve, same resamples
    runs={k:{s:matched(tables[solver],idx,s,[cfg(solver,t) for t in taus],taus) for s in SAMPLERS} for k,(solver,taus) in CURVES.items()}
    match={}
    for k,per in runs.items():
        match[k]={}
        for s,r in per.items():
            e={'tau_eff':ci(r['tau_eff']),'in_range':bool(r['in_range'][0])}
            for m in ('ade1','flip'):
                lo,hi=np.quantile(r[m]['residual'][1:],[.05,.95])
                e[m]={'reference':float(r[m]['reference'][0]),'sampler':float(r[m]['sampler'][0]),'matched':float(r[m]['matched'][0]),
                    'effect':ci(r[m]['effect']),'residual':ci(r[m]['residual']),'tost90':[float(lo),float(hi)],
                    'equivalent':bool(-MARGINS[m]<lo and hi<MARGINS[m])}
            match[k][s]=e
    result['matched']=match
    pair={}
    for s in SAMPLERS:
        for a,b in itertools.combinations(list(CURVES),2):
            pair.setdefault(s,{})[f'{a}-{b}']={'tau_eff':ci(runs[a][s]['tau_eff']-runs[b][s]['tau_eff']),
                **{f'{m}_residual':ci(runs[a][s][m]['residual']-runs[b][s][m]['residual']) for m in ('ade1','flip')}}
    result['pairwise_differences']=pair
    e10=lambda k:match[k]['euler10_t1.0'];three=['heun32','heun64','heun128']
    d=max(abs(e10(a)['tau_eff'][0]-e10(b)['tau_eff'][0]) for a,b in itertools.combinations(three,2))
    result['C2']={'tau_eff':{k:e10(k)['tau_eff'] for k in three},'max_pairwise_abs_difference':d,'holds':bool(d<.02)}
    verdicts={k:{m:e10(k)[m]['equivalent'] for m in ('ade1','flip')} for k in three+['heun32_dense']}
    result['C3']={'verdicts':verdicts,'holds':bool(len({tuple(v.values()) for v in verdicts.values()})==1)}
    expl={k:{m:e10(k)[m]['equivalent'] for m in ('ade1','flip')} for k in ('heun32_on_a18_grid','heun64_step01','heun128_step01')}
    result['exploratory_C3_same_grid_verdicts']=expl
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'results.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(f"== Addendum 18, {len(names)} clips, {args.boot} resamples seed {args.seed}")
    print('1. Heun-32 - Heun-128 (and Heun-64 - Heun-128, exploratory):')
    for a in ('heun32','heun64'):
        for t,r in disc[f'{a}_minus_heun128'].items():
            if t=='exploratory':continue
            print(f"  {a:7s} tau {t:5s} spread {r['spread'][0]:+.4f} [{r['spread'][1]:+.4f},{r['spread'][2]:+.4f}]"
                  f" ade1 {r['ade1'][0]:+.4f} [{r['ade1'][1]:+.4f},{r['ade1'][2]:+.4f}] flip {100*r['flip'][0]:+.2f} [{100*r['flip'][1]:+.2f},{100*r['flip'][2]:+.2f}] pt")
    print('  C1',result['C1'])
    print('2./3. matched temperature:')
    for k in CURVES:
        for s in SAMPLERS:
            e=match[k][s]
            print(f"  {k:19s} {s:12s} tau {e['tau_eff'][0]:.3f} [{e['tau_eff'][1]:.3f},{e['tau_eff'][2]:.3f}]{'' if e['in_range'] else ' OUT'}"
                  f" | ade1 res {e['ade1']['residual'][0]:+.4f} tost {e['ade1']['tost90'][0]:+.4f},{e['ade1']['tost90'][1]:+.4f} eq {e['ade1']['equivalent']}"
                  f" | flip res {100*e['flip']['residual'][0]:+.2f} tost {100*e['flip']['tost90'][0]:+.2f},{100*e['flip']['tost90'][1]:+.2f} eq {e['flip']['equivalent']}")
    print('  C2',result['C2']);print('  C3',result['C3']);print('  exploratory same-grid verdicts',expl)


if __name__=='__main__':main()
