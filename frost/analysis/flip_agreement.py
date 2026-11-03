"""EXPLORATORY (not pre-registered): do E10 and the cold reference flip the SAME decisions?

The paper's flip residual compares two RATES of disagreement with Heun-32 tau=1 (E10's and the reference's at
tau_eff). Equal rates can come from different clips/classes. Here, on the 400 held-out PhysicalAI-AV clips and
under the same initial noise z (same seed), we compare the decisions themselves.

The reference exists only on the tau grid {1, .9, ..., .4}, so the "cold reference" is each grid neighbour of
E10's tau_eff (lo = floor, hi = ceil to 0.1), reported separately; for transition matrices we also report the
linear mixture w*lo + (1-w)*hi with w = (hi - tau_eff)/0.1 (the same linear interpolation the paper uses).

Per model:
(a) 2x2 table over clip x seed: "E10 flips vs ref tau=1" x "cold ref flips vs ref tau=1"; overlap statistics.
(b) agreement of E10's decision with the cold reference's decision (same seed), against
    - two reference temperatures 0.1 apart, same seed (the pair bracketing tau_eff, and other adjacent pairs),
    - a linear "same-distance" expectation 1 - (d/0.1)(1 - agree_0.1), d = |tau_eff - neighbour| (heuristic),
    - the floor: ref tau=1 vs itself across different seeds (independent noise).
(c) transition matrices (longitudinal, lateral) from ref tau=1 to E10 and to the cold reference: joint
    probabilities P(ref class i, X class j); total variation (TV) between E10's and the cold reference's matrices,
    over all cells and over the off-diagonal (flip) cells renormalized. TV of empirical matrices is biased up;
    TV(cold lo, cold hi) is given as a scale.
Decision = (longitudinal, lateral) of frost.core.decisions.decisions with the clip's speed_mps,
as in frost.analysis.paper_numbers. Clip bootstrap (default 2000, seed 20261030), clips resampled jointly across models.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions

LONG=['stop','decelerate','keep','accelerate'];LAT=['left','shift_left','straight','shift_right','right']
GRID=[1.,.9,.8,.7,.6,.5,.4]
MODELS=[('alpamayo15','a15'),('alpamayo_r1','r1'),('alpamayo2','a2')]
SAMPLER='euler10_t1.0'


def cfg(t):return f'heun32_t{t:.1f}'


def load_classes(probe,names,records,cfgs):
    """Decision class indices: dict cfg -> (long, lat) int arrays of shape (clips, seeds)."""
    seeds=json.loads((probe/'manifest.json').read_text())['seeds'];out={}
    with np.load(probe/'plans.npz') as z:
        for c in cfgs:
            lo=np.zeros((len(names),len(seeds)),int);la=np.zeros_like(lo)
            for i,n in enumerate(names):
                v=records[n]['speed_mps']
                for j,s in enumerate(seeds):
                    d=decisions(z[f'{n}_s{s}_{c}'],v);lo[i,j]=LONG.index(d['longitudinal']);la[i,j]=LAT.index(d['lateral'])
            out[c]=(lo,la)
    return seeds,out


def joint(c):return c[0]*len(LAT)+c[1]


def per_clip(cls,lo_t,hi_t):
    """Per-clip sufficient statistics (sums over seeds), so the bootstrap is a weighted sum over clips."""
    ref=cls[cfg(1.)];e10=cls[SAMPLER];S=ref[0].shape[1];st={}
    for name,t in [('lo',lo_t),('hi',hi_t)]:
        cold=cls[cfg(t)];E=joint(e10)!=joint(ref);C=joint(cold)!=joint(ref)
        st[f'{name}_EC']=np.stack([(E&C).sum(1),(E&~C).sum(1),(~E&C).sum(1),(~E&~C).sum(1)],1)
        st[f'{name}_same_flip']=(E&C&(joint(e10)==joint(cold))).sum(1)
        for ax,k in [('joint',None),('long',0),('lat',1)]:
            f=(lambda c:joint(c)) if k is None else (lambda c,k=k:c[k])
            st[f'agree_e10_{name}_{ax}']=(f(e10)==f(cold)).sum(1)
    pairs=[(a,b) for a,b in zip(GRID[:-1],GRID[1:])]
    for a,b in pairs:
        for ax,k in [('joint',None),('long',0),('lat',1)]:
            f=(lambda c:joint(c)) if k is None else (lambda c,k=k:c[k])
            st[f'agree_ref_{a:.1f}_{b:.1f}_{ax}']=(f(cls[cfg(a)])==f(cls[cfg(b)])).sum(1)
    for ax,k in [('joint',None),('long',0),('lat',1)]:
        f=(lambda c:joint(c)) if k is None else (lambda c,k=k:c[k]);r=f(ref)
        st[f'agree_e10_ref1_{ax}']=(f(e10)==r).sum(1)
        st[f'floor_{ax}']=np.stack([(r[:,i]==r[:,j]) for i,j in itertools.combinations(range(S),2)],1).sum(1)
    for ax,k,K in [('long',0,len(LONG)),('lat',1,len(LAT))]:
        for nm,c in [('e10',e10),('lo',cls[cfg(lo_t)]),('hi',cls[cfg(hi_t)])]:
            idx=ref[k]*K+c[k];st[f'T_{ax}_{nm}']=np.stack([np.bincount(row,minlength=K*K) for row in idx])
    return st,S


def tv(p,q):return .5*np.abs(p-q).sum()


def offdiag(P,K):
    m=P.reshape(K,K).copy();np.fill_diagonal(m,0);s=m.sum();return m.ravel()/s if s>0 else m.ravel()


def summarize(st,S,w_lo,tau,lo_t,hi_t,weights):
    """All statistics for one clip weighting (weights = clip multiplicities; ones = point estimate)."""
    W=lambda k:weights@st[k];out={};n=weights.sum()*S
    for name in ['lo','hi']:
        a,b,c,d=W(f'{name}_EC');both=a;e=a+b;cc=a+c
        out[name]={'table':{'both':both,'e10_only':b,'cold_only':c,'neither':d},'e10_flip':e/n,'cold_flip':cc/n,
            'p_cold_given_e10':both/e if e else np.nan,'p_e10_given_cold':both/cc if cc else np.nan,
            'jaccard':both/(a+b+c) if a+b+c else np.nan,'both_expected_indep':e*cc/n,
            'same_class_given_both':W(f'{name}_same_flip')/both if both else np.nan}
        for ax in ['joint','long','lat']:out[name][f'agree_{ax}']=W(f'agree_e10_{name}_{ax}')/n
    for ax in ['joint','long','lat']:
        out[f'agree_e10_ref1_{ax}']=W(f'agree_e10_ref1_{ax}')/n
        out[f'floor_{ax}']=W(f'floor_{ax}')/(weights.sum()*S*(S-1)/2)
        for a,b in zip(GRID[:-1],GRID[1:]):out[f'agree_ref_{a:.1f}_{b:.1f}_{ax}']=W(f'agree_ref_{a:.1f}_{b:.1f}_{ax}')/n
        bracket=out[f'agree_ref_{hi_t:.1f}_{lo_t:.1f}_{ax}']
        for name,t in [('lo',lo_t),('hi',hi_t)]:
            d=abs(tau-t);out[name][f'expected_same_distance_{ax}']=1-(d/.1)*(1-bracket)
            out[name][f'agree_minus_bracket_{ax}']=out[name][f'agree_{ax}']-bracket
    for ax,K in [('long',len(LONG)),('lat',len(LAT))]:
        P={nm:W(f'T_{ax}_{nm}')/n for nm in ['e10','lo','hi']};P['mix']=w_lo*P['lo']+(1-w_lo)*P['hi']
        o={f'tv_e10_{nm}':tv(P['e10'],P[nm]) for nm in ['lo','hi','mix']};o['tv_lo_hi']=tv(P['lo'],P['hi'])
        o.update({f'tv_offdiag_e10_{nm}':tv(offdiag(P['e10'],K),offdiag(P[nm],K)) for nm in ['lo','hi','mix']})
        o['tv_offdiag_lo_hi']=tv(offdiag(P['lo'],K),offdiag(P['hi'],K))
        o['matrices']={nm:P[nm].reshape(K,K) for nm in P}
        out[f'transition_{ax}']=o
    return out


def flatten(d,prefix=''):
    for k,v in d.items():
        if isinstance(v,dict):yield from flatten(v,f'{prefix}{k}.')
        elif np.ndim(v)==0:yield f'{prefix}{k}',float(v)


def tolist(d):
    if isinstance(d,dict):return {k:tolist(v) for k,v in d.items()}
    if isinstance(d,np.ndarray):return d.round(6).tolist()
    if isinstance(d,(np.floating,np.integer)):return float(d)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--probe-pattern',default='temperature_probe_{tag}_heldout_pooled400_v3_20261028')
    p.add_argument('--numbers',type=Path,default=Path('artifacts/paper_numbers_heldout_pooled400_v2_20261028/summary.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261030)
    args=p.parse_args()
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    names=sorted(records);numbers=json.loads(args.numbers.read_text())
    rng=np.random.default_rng(args.seed);W=np.stack([np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names)) for _ in range(args.boot)])
    result={'status':'EXPLORATORY, not pre-registered','clips':len(names),'boot':args.boot,'boot_seed':args.seed,
        'classes':{'long':LONG,'lat':LAT},'sampler':SAMPLER,'reference':cfg(1.),'models':{}}
    for key,tag in MODELS:
        tau=numbers['models'][key][SAMPLER]['tau_eff'][0];lo_t=np.floor(tau*10+1e-9)/10;hi_t=np.ceil(tau*10-1e-9)/10
        w_lo=(hi_t-tau)/.1
        seeds,cls=load_classes(args.artifacts/args.probe_pattern.format(tag=tag),names,records,[SAMPLER]+[cfg(t) for t in GRID])
        st,S=per_clip(cls,lo_t,hi_t)
        point=summarize(st,S,w_lo,tau,lo_t,hi_t,np.ones(len(names)))
        flat=dict(flatten({k:v for k,v in point.items()}));boots={k:[] for k in flat}
        for w in W:
            for k,v in flatten(summarize(st,S,w_lo,tau,lo_t,hi_t,w)):boots[k].append(v)
        ci={k:[float(np.nanquantile(v,.025)),float(np.nanquantile(v,.975))] for k,v in boots.items()}
        # Distinct clips behind the flips (point estimate).
        ref=joint(cls[cfg(1.)]);E=joint(cls[SAMPLER])!=ref
        distinct={'e10_any_flip':int(E.any(1).sum())}
        for nm,t in [('lo',lo_t),('hi',hi_t)]:
            C=joint(cls[cfg(t)])!=ref;distinct[f'cold_{nm}_any_flip']=int(C.any(1).sum());distinct[f'both_{nm}_same_seed']=int((E&C).any(1).sum())
        result['models'][key]={'tau_eff':tau,'neighbours':{'lo':lo_t,'hi':hi_t},'w_lo_mix':w_lo,'seeds':seeds,
            'point':tolist(point),'ci95':ci,'distinct_clips':distinct}
        print(key,f'tau_eff {tau:.3f} lo {lo_t} hi {hi_t}',flush=True)
        for nm in ['lo','hi']:
            q=point[nm];print(f"  {nm}: table {dict((k,int(v)) for k,v in q['table'].items())} P(C|E) {q['p_cold_given_e10']:.2f} P(E|C) {q['p_e10_given_cold']:.2f} "
                f"agree {q['agree_joint']:.3f} [{ci[nm+'.agree_joint'][0]:.3f},{ci[nm+'.agree_joint'][1]:.3f}] exp-same-dist {q['expected_same_distance_joint']:.3f} "
                f"minus bracket {q['agree_minus_bracket_joint']:+.3f} [{ci[nm+'.agree_minus_bracket_joint'][0]:+.3f},{ci[nm+'.agree_minus_bracket_joint'][1]:+.3f}]")
        print(f"  bracket {point[f'agree_ref_{hi_t:.1f}_{lo_t:.1f}_joint']:.3f} floor {point['floor_joint']:.3f} e10-vs-ref1 {point['agree_e10_ref1_joint']:.3f}")
        for ax in ['long','lat']:
            o=point[f'transition_{ax}'];print(f"  {ax}: TV e10-mix {o['tv_e10_mix']:.4f} [{ci[f'transition_{ax}.tv_e10_mix'][0]:.4f},{ci[f'transition_{ax}.tv_e10_mix'][1]:.4f}] "
                f"lo {o['tv_e10_lo']:.4f} hi {o['tv_e10_hi']:.4f} lo-hi {o['tv_lo_hi']:.4f} | offdiag mix {o['tv_offdiag_e10_mix']:.3f} [{ci[f'transition_{ax}.tv_offdiag_e10_mix'][0]:.3f},{ci[f'transition_{ax}.tv_offdiag_e10_mix'][1]:.3f}] lo-hi {o['tv_offdiag_lo_hi']:.3f}")
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=1))
    (args.out/'source_snapshot.py').write_text(Path(__file__).read_text())


if __name__=='__main__':
    main()
