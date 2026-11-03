"""Addendum 16: extended temperature curve (Heun-32 at tau 1..0.4 from the main run, 0.3/0.2/0.1 from the low-tau run).

For E1, E2, E4 of the main run: tau_eff by spread matching on the extended curve and ADE_1 / flip residuals against it;
flips are against the main run's Heun-32 tau=1 reference of the same seed (the low-tau run reuses the same seeds and
initial noise). Clip bootstrap (2000 draws, seed 20261029) re-solving tau_eff; 90% intervals decide equivalence at the
protocol margins, 95% reported. Predictions L1 (E2 tau_eff in [0.2, 0.4]), L2 (E1 tau_eff >= 0.1, in range) and
L3 (exploratory: E2 flip residual > +2 pt) are evaluated per model; L1/L3 are judged across models by the caller.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.paper_numbers import REFERENCE,MARGINS,clip_metrics

TAUS=[1.,.9,.8,.7,.6,.5,.4,.3,.2,.1];CURVE=[f'heun32_t{t}' for t in TAUS];SAMPLERS=['euler1_t1.0','euler2_t1.0','euler4_t1.0']


def matched(table,sample,sampler):
    M=lambda c,m:float(np.mean([table[c][n][m] for n in sample]))
    sp=np.array([M(c,'spread') for c in CURVE]);o=np.argsort(sp);x=M(sampler,'spread')
    out={'tau_eff':float(np.interp(x,sp[o],np.array(TAUS)[o])),'in_range':bool(sp.min()<=x<=sp.max()),'spread':x}
    for m in ['ade1','flip']:
        at=float(np.interp(x,sp[o],np.array([M(c,m) for c in CURVE])[o]));v=M(sampler,m)
        out[m]={'sampler':v,'matched':at,'residual':v-at,'effect':v-M(REFERENCE,m)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--main',type=Path,required=True)
    p.add_argument('--lowtau',type=Path,required=True);p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--boot',type=int,default=2000);a=p.parse_args()
    records={r['name']:r for r in json.loads((a.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*a.range)];seeds=json.loads((a.main/'manifest.json').read_text())['seeds']
    plans={}
    for path in (a.main,a.lowtau):
        with np.load(path/'plans.npz') as z:plans.update({k:z[k] for k in z.files if k.split('_',3)[-1] in CURVE+SAMPLERS})
    table=collections.defaultdict(dict)
    for n in names:
        human=np.load(a.data/f'{n}.npz')['ego_future_xyz'];speed=records[n]['speed_mps']
        ref=[decisions(plans[f'{n}_s{s}_{REFERENCE}'],speed) for s in seeds]
        for c in CURVE+SAMPLERS:table[c][n]=clip_metrics(plans,n,c,seeds,speed,human,ref)
    M=lambda c,m:float(np.mean([table[c][n][m] for n in names]))
    res={'clips':len(names),'margins':MARGINS,'curve':{c:{m:M(c,m) for m in ['spread','ade1','flip']} for c in CURVE},'samplers':{}}
    rng=np.random.default_rng(20261029);draws=[rng.choice(names,len(names)) for _ in range(a.boot)]
    for s in SAMPLERS:
        m=matched(table,names,s);B=[matched(table,d,s) for d in draws]
        m['tau_eff_ci95']=np.quantile([b['tau_eff'] for b in B],[.025,.975]).tolist();m['boot_in_range']=float(np.mean([b['in_range'] for b in B]))
        for q in ['ade1','flip']:
            r=[b[q]['residual'] for b in B];m[q]['residual_ci95']=np.quantile(r,[.025,.975]).tolist();m[q]['residual_ci90']=np.quantile(r,[.05,.95]).tolist()
            m[q]['equivalent']=bool(m['in_range'] and -MARGINS[q]<m[q]['residual_ci90'][0] and m[q]['residual_ci90'][1]<MARGINS[q])
        res['samplers'][s]=m
    e1,e2=res['samplers']['euler1_t1.0'],res['samplers']['euler2_t1.0']
    res['L1']=bool(e2['in_range'] and .2<=e2['tau_eff']<=.4);res['L2']=bool(e1['in_range'] and e1['tau_eff']>=.1);res['L3']=bool(e2['flip']['residual']>.02)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1));(a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print({c:round(v['spread'],3) for c,v in res['curve'].items()})
    for s,m in res['samplers'].items():
        print(f"{s}: spread {m['spread']:.3f} tau {m['tau_eff']:.3f} {[round(x,3) for x in m['tau_eff_ci95']]} in_range {m['in_range']} | ADE resid {m['ade1']['residual']:+.3f} {[round(x,3) for x in m['ade1']['residual_ci95']]} eq {m['ade1']['equivalent']} | flip resid {100*m['flip']['residual']:+.1f} {[round(100*x,1) for x in m['flip']['residual_ci95']]} eq {m['flip']['equivalent']}")
    print('L1',res['L1'],'L2',res['L2'],'L3',res['L3'])


if __name__=='__main__':main()
