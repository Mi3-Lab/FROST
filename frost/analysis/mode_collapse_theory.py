"""Direction of numerical decision flips in solvable K-mode Gaussian-mixture flows.

Tests whether under-resolved integration moves decisions only from less to more
probable modes (numerical mode collapse), across solvers, schedules, mode counts
and geometries. Closed-form marginal velocity for x_t=(1-t)z+t x1.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def velocity(x,t,means,weights,s2):
    var=(1-t)**2+t**2*s2;diff=x[:,None,:]-t*means[None]
    logits=np.log(weights)[None]-np.sum(diff**2,-1)/(2*var);logits-=logits.max(1,keepdims=True)
    post=np.exp(logits);post/=post.sum(1,keepdims=True)
    gain=(t*s2-(1-t))/var
    return post@means+gain*(x-t*(post@means))


def solve(z,means,weights,s2,grid,method):
    x=z.copy();v=None
    for i,(a,b) in enumerate(zip(grid[:-1],grid[1:])):
        h=b-a
        if method=='cached' and 2<=i<len(grid)-3 and v is not None:pass
        else:v=velocity(x,a,means,weights,s2)
        if method=='heun':x=x+.5*h*(v+velocity(x+h*v,b,means,weights,s2))
        elif method=='rk4':
            k2=velocity(x+.5*h*v,a+.5*h,means,weights,s2);k3=velocity(x+.5*h*k2,a+.5*h,means,weights,s2);k4=velocity(x+h*k3,b,means,weights,s2)
            x=x+h*(v+2*k2+2*k3+k4)/6
        else:x=x+h*v
    return x


def classify(x,means):return np.argmin(np.sum((x[:,None]-means[None])**2,-1),1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    rng=np.random.default_rng(31);results=[]
    configs=[]
    for k in [2,3,5]:
        for dim in [2,16]:
            for scale in [1.,3.]:
                for trial in range(3):
                    weights=rng.dirichlet(np.ones(k)*1.5);means=rng.standard_normal((k,dim))*scale
                    configs.append((k,dim,scale,weights,means))
    up=down=same=0
    for k,dim,scale,weights,means in configs:
        z=rng.standard_normal((20000,dim));exact=classify(solve(z,means,weights,.05,np.linspace(0,1,401),'rk4'),means)
        order=np.argsort(np.argsort(-weights))  # rank 0 = most probable mode
        entry={'k':k,'dim':dim,'scale':scale,'weights':weights.round(4).tolist(),'solvers':{}}
        for method,steps in [('euler',2),('euler',4),('euler',8),('heun',2),('heun',4),('rk4',1),('rk4',2),('cached',8)]:
            c=classify(solve(z,means,weights,.05,np.linspace(0,1,steps+1),method),means);flip=c!=exact
            to_more=int((order[c[flip]]<order[exact[flip]]).sum());to_less=int((order[c[flip]]>order[exact[flip]]).sum())
            freq_exact=np.bincount(exact,minlength=k)/len(z);freq=np.bincount(c,minlength=k)/len(z)
            ent=lambda q:float(-(q[q>0]*np.log2(q[q>0])).sum())
            entry['solvers'][f'{method}{steps}']={'flips':int(flip.sum()),'to_more_probable':to_more,'to_less_probable':to_less,
                'entropy_change_bits':ent(freq)-ent(freq_exact),'rarest_mode_freq_exact':float(freq_exact[np.argmin(weights)]),'rarest_mode_freq':float(freq[np.argmin(weights)])}
            up+=to_more;down+=to_less
        results.append(entry)
        s=entry['solvers'];print(json.dumps({'k':k,'dim':dim,'scale':scale,'w':entry['weights'],**{m:(v['to_more_probable'],v['to_less_probable'],round(v['entropy_change_bits'],3)) for m,v in s.items()}}),flush=True)
    summary={'flips_to_more_probable':up,'flips_to_less_probable':down,'configs':len(configs),
        'entropy_decreased_fraction':float(np.mean([v['entropy_change_bits']<0 for r in results for v in r['solvers'].values() if v['flips']>0]))}
    print(json.dumps(summary))
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'summary':summary,'results':results},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
