"""Exploratory: is the one/two-step ADE_1 gain a mean-regression (Jensen) effect?

Baseline: per clip, the mean of the six reference samples at each tau in 1..0.1 (a point estimate of the conditional
mean); its ADE_1 is minimised over tau. Compared with E_k single-sample ADE_1 and with E_k's own six-sample mean.
Clip bootstrap (2000 draws, seed 20261029), minimum over tau re-taken in every resample. CPU only; not registered.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis.lowtau_analysis import TAUS,CURVE,SAMPLERS


def ade(t,h):return float(np.linalg.norm(t[:,:2]-h[:,:2],axis=1).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--main',type=Path,required=True)
    p.add_argument('--lowtau',type=Path,required=True);p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--boot',type=int,default=2000);a=p.parse_args()
    names=[f'context_{i:03d}' for i in range(*a.range)];seeds=json.loads((a.main/'manifest.json').read_text())['seeds']
    plans={}
    for path in (a.main,a.lowtau):
        with np.load(path/'plans.npz') as z:plans.update({k:z[k] for k in z.files if k.split('_',3)[-1] in CURVE+SAMPLERS})
    human={n:np.load(a.data/f'{n}.npz')['ego_future_xyz'] for n in names}
    T=lambda n,c:[plans[f'{n}_s{s}_{c}'] for s in seeds]
    single={c:np.array([np.mean([ade(t,human[n]) for t in T(n,c)]) for n in names]) for c in CURVE+SAMPLERS}
    meanade={c:np.array([ade(np.mean(T(n,c),0),human[n]) for n in names]) for c in CURVE+SAMPLERS}
    M=np.array([meanade[c] for c in CURVE])
    def stat(ix,x):
        m=M[:,ix].mean(1);j=int(np.argmin(m));return float(x[ix].mean()-m[j]),float(m[j]),TAUS[j]
    rng=np.random.default_rng(20261029);draws=[rng.integers(0,len(names),len(names)) for _ in range(a.boot)];full=np.arange(len(names))
    res={'clips':len(names),'reference_single_ade1':{c:float(single[c].mean()) for c in CURVE},'reference_mean6_ade1':{c:float(meanade[c].mean()) for c in CURVE},'samplers':{}}
    for s in SAMPLERS:
        out={'single_ade1':float(single[s].mean()),'mean6_ade1':float(meanade[s].mean())}
        for k,x in (('single_minus_best_ref_mean6',single[s]),('mean6_minus_best_ref_mean6',meanade[s])):
            d,best,tau=stat(full,x);B=np.array([stat(ix,x)[0] for ix in draws])
            out[k]={'diff':d,'best_ref_mean6_ade1':best,'best_tau':tau,'ci95':np.quantile(B,[.025,.975]).tolist(),'frac_below_zero':float(np.mean(B<0))}
        res['samplers'][s]=out;print(s,json.dumps(out))
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=1));(a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
