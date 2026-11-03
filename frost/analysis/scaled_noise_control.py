"""Addendum 17: scaled-noise Euler-10 control (the test of Yang et al. 2610.32102) on A1.5 held-out clips.

Two temperature curves from the same probe: (A) Heun-32 at tau (ours) and (B) Euler-10 at tau (scaled noise, Yang).
For E2, E4, E6: tau_eff by spread matching and ADE_1 / flip residuals against each curve; flips against the Heun-32
tau=1 reference of the same seed. Clip bootstrap (2000 draws, seed 20261029) re-solving tau_eff; 90% intervals decide
equivalence at the protocol margins, 95% reported. Curve B also reports the unmatched comparison at tau=1/k (Y3).
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.paper_numbers import TAUS,REFERENCE,MARGINS,clip_metrics

CURVES={'A_heun32':[f'heun32_t{t}' for t in TAUS],'B_euler10':['euler10_t1.0']+[f'euler10_t{t}' for t in TAUS[1:]]}
SAMPLERS=['euler2_t1.0','euler4_t1.0','euler6_t1.0']


def matched(table,sample,curve,sampler):
    M=lambda c,m:float(np.mean([table[c][n][m] for n in sample]))
    sp=np.array([M(c,'spread') for c in curve]);o=np.argsort(sp);x=M(sampler,'spread')
    out={'tau_eff':float(np.interp(x,sp[o],np.array(TAUS)[o])),'in_range':bool(sp.min()<=x<=sp.max()),'spread':x}
    for m in ['ade1','flip']:
        at=float(np.interp(x,sp[o],np.array([M(c,m) for c in curve])[o]));v=M(sampler,m)
        out[m]={'sampler':v,'matched':at,'residual':v-at,'effect':v-M(REFERENCE,m)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--out',type=Path,required=True);p.add_argument('--boot',type=int,default=2000);a=p.parse_args()
    records={r['name']:r for r in json.loads((a.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*a.range)];seeds=json.loads((a.probe/'manifest.json').read_text())['seeds']
    with np.load(a.probe/'plans.npz') as z:plans={k:z[k] for k in z.files}
    configs=sorted(set(SAMPLERS+[c for v in CURVES.values() for c in v]));table=collections.defaultdict(dict)
    for n in names:
        human=np.load(a.data/f'{n}.npz')['ego_future_xyz'];speed=records[n]['speed_mps']
        ref=[decisions(plans[f'{n}_s{s}_{REFERENCE}'],speed) for s in seeds]
        for c in configs:table[c][n]=clip_metrics(plans,n,c,seeds,speed,human,ref)
    M=lambda c,m,S=names:float(np.mean([table[c][n][m] for n in S]))
    res={'clips':len(names),'margins':MARGINS,'curves':{k:{c:{m:M(c,m) for m in ['spread','ade1','flip']} for c in v} for k,v in CURVES.items()},'samplers':{}}
    rng=np.random.default_rng(20261029);draws=[rng.choice(names,len(names)) for _ in range(a.boot)]
    for s in SAMPLERS:
        res['samplers'][s]={}
        for k,curve in CURVES.items():
            m=matched(table,names,curve,s);B=[matched(table,d,curve,s) for d in draws]
            m['tau_eff_ci95']=np.quantile([b['tau_eff'] for b in B],[.025,.975]).tolist()
            for q in ['ade1','flip']:
                r=[b[q]['residual'] for b in B];m[q]['residual_ci95']=np.quantile(r,[.025,.975]).tolist();m[q]['residual_ci90']=np.quantile(r,[.05,.95]).tolist()
                m[q]['equivalent']=bool(m['in_range'] and -MARGINS[q]<m[q]['residual_ci90'][0] and m[q]['residual_ci90'][1]<MARGINS[q])
            res['samplers'][s][k]=m
        k_steps=int(s[5:s.index('_')]);t=min(TAUS,key=lambda x:abs(x-1/k_steps));c=f'euler10_t{t}'
        res['samplers'][s]['Y3_unmatched_euler10_tau_1_over_k']={'config':c,**{q:M(s,q)-M(c,q) for q in ['spread','ade1','flip']}}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1));(a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for k,v in res['curves'].items():print(k,{c:round(x['spread'],3) for c,x in v.items()})
    for s,v in res['samplers'].items():
        for k in CURVES:
            m=v[k];print(f"{s} vs {k}: tau {m['tau_eff']:.3f} {[round(x,3) for x in m['tau_eff_ci95']]} in_range {m['in_range']} | ADE resid {m['ade1']['residual']:+.3f} {[round(x,3) for x in m['ade1']['residual_ci95']]} eq {m['ade1']['equivalent']} | flip resid {100*m['flip']['residual']:+.1f} {[round(100*x,1) for x in m['flip']['residual_ci95']]} eq {m['flip']['equivalent']}")
        print(s,'Y3',v['Y3_unmatched_euler10_tau_1_over_k'])


if __name__=='__main__':main()
