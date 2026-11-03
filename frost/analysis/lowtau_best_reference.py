"""Exploratory: E_k single-sample ADE_1 minus the best (minimum over tau in 1..0.1) reference ADE_1.

Reuses frost.analysis.lowtau_analysis (curve, samplers) and frost.analysis.paper_numbers.clip_metrics on the same inputs. Clip bootstrap
(2000 draws, seed 20261029); the minimum over tau is re-taken in every resample. CPU only; not a registered prediction.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.paper_numbers import REFERENCE,clip_metrics
from frost.analysis.lowtau_analysis import TAUS,CURVE,SAMPLERS


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
    idx={n:i for i,n in enumerate(names)}
    A=np.array([[table[c][n]['ade1'] for n in names] for c in CURVE]);E={s:np.array([table[s][n]['ade1'] for n in names]) for s in SAMPLERS}
    def stat(ix,s):
        m=A[:,ix].mean(1);j=int(np.argmin(m));return float(E[s][ix].mean()-m[j]),float(m[j]),TAUS[j]
    rng=np.random.default_rng(20261029);draws=[rng.integers(0,len(names),len(names)) for _ in range(a.boot)]
    full=np.arange(len(names));res={'clips':len(names),'curve_ade1':{c:float(A[i].mean()) for i,c in enumerate(CURVE)},'samplers':{}}
    for s in SAMPLERS:
        d,best,tau=stat(full,s);B=np.array([stat(ix,s)[0] for ix in draws])
        res['samplers'][s]={'ade1':float(E[s].mean()),'best_reference_ade1':best,'best_reference_tau':tau,'diff':d,
            'diff_ci95':np.quantile(B,[.025,.975]).tolist(),'diff_ci90':np.quantile(B,[.05,.95]).tolist(),'frac_below_zero':float(np.mean(B<0))}
        print(s,res['samplers'][s])
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1))


if __name__=='__main__':main()
