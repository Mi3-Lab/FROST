"""Parameter-free prediction of Euler contraction from the policy's conditional action variance.

For a Gaussian target N(m, s^2) under rectified flow, Euler-k shrinks the deviation from m
by c_k(s) = prod_i cos(angle(u(t_i), u(t_{i+1}))), u(t) = (1-t, t s). We estimate the
per-dimension conditional standard deviation s_d of each clip from two converged samples
(Heun160, shared language trace) and predict the spread ratio sqrt(sum s_d^2 c_k^2 / sum s_d^2).
The measured ratio is the distance to the conditional mean (one Euler step) of the Euler-k
sample over that of the converged sample from the same noise.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def contraction(s,k):
    s=np.asarray(s,float);t=np.linspace(0,1,k+1);ratio=np.ones_like(s)
    for i in range(k):
        a0,a1=1-t[i],t[i]*s;b0,b1=1-t[i+1],t[i+1]*s
        ratio=ratio*(a0*b0+a1*b1)/np.sqrt(a0**2+a1**2)/np.sqrt(b0**2+b1**2)
    return ratio


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    z=np.load(args.predictions/'predictions.npz');names=sorted({k.split('_s92')[0] for k in z.files if '_s92' in k})
    std=np.array([np.abs(z[f'{n}_s9201_heun160_action'].ravel()-z[f'{n}_s9202_heun160_action'].ravel())/np.sqrt(2) for n in names])
    result={'clips':len(names),'conditional_std_quartiles':np.quantile(std,[.25,.5,.75]).tolist(),'steps':{}}
    for k in [4,8,10]:
        c=contraction(std,k);predicted=np.sqrt((std**2*c**2).sum(1)/(std**2).sum(1));measured=[]
        for n in names:
            for seed in ['9201','9202']:
                mean=(z[f'{n}_s{seed}_euler10_states'][0]+z[f'{n}_s{seed}_euler10_velocities'][0]).ravel()
                measured.append(np.linalg.norm(z[f'{n}_s{seed}_euler{k}_action'].ravel()-mean)/np.linalg.norm(z[f'{n}_s{seed}_heun160_action'].ravel()-mean))
        result['steps'][k]={'predicted_median':float(np.median(predicted)),'predicted_iqr':np.quantile(predicted,[.25,.75]).tolist(),
            'measured_median':float(np.median(measured)),'measured_iqr':np.quantile(measured,[.25,.75]).tolist(),
            'fraction_contracted':float(np.mean(np.array(measured)<1))}
    result['curve']={str(s):{'c4':float(contraction(s,4)),'c10':float(contraction(s,10))} for s in [.05,.1,.3,.5,1,2]}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result['steps'],indent=1))


if __name__=='__main__':main()
