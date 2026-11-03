"""Clip-clustered statistics for comparing samplers at matched effective temperature.

For each model probe and each Euler-k sampler: effective temperature from seed spread,
effect against the tau=1 reference, residual against the explicit-temperature curve
interpolated at tau_eff, fraction of the effect explained by temperature, and a TOST-style
equivalence check of the residual against declared margins. Every quantity is
recomputed inside a bootstrap over clips (clips are the independent unit; seeds within
a clip are not).
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.core.temperature_report import entropy

METRICS=['ade1','minade6','flip','entropy','stop','decelerate']
# Equivalence margins declared before computing residuals (26/10/2026).
MARGINS={'ade1':.05,'minade6':.05,'flip':.02,'entropy':.05,'stop':.01,'decelerate':.02}


def per_clip(plans,names,configs,seeds,reference,human,speeds):
    out=collections.defaultdict(dict)
    for n in names:
        ref=[decisions(plans[f'{n}_s{s}_{reference}'],speeds[n]) for s in seeds];key=lambda d:(d['longitudinal'],d['lateral'])
        for c in configs:
            trajs=[plans[f'{n}_s{s}_{c}'] for s in seeds];d=[decisions(t,speeds[n]) for t in trajs]
            e=[np.linalg.norm(t[:,:2]-human[n][:,:2],axis=1).mean() for t in trajs];lon=[x['longitudinal'] for x in d]
            out[c][n]={'spread':float(np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]])),
                'ade1':float(np.mean(e)),'minade6':float(min(e)),'flip':float(np.mean([key(a)!=key(b) for a,b in zip(d,ref)])),
                'entropy':entropy([key(x) for x in d]),'stop':float(np.mean([x=='stop' for x in lon])),'decelerate':float(np.mean([x=='decelerate' for x in lon]))}
    return out


def summarize(table,sample,curve_configs,curve_taus,sampler):
    mean=lambda c,m:float(np.mean([table[c][n][m] for n in sample]))
    spreads=np.array([mean(c,'spread') for c in curve_configs]);order=np.argsort(spreads);taus=np.array(curve_taus)
    s=mean(sampler,'spread');tau=float(np.interp(s,spreads[order],taus[order]));res={'tau_eff':tau}
    for m in METRICS:
        curve=np.array([mean(c,m) for c in curve_configs]);at=float(np.interp(s,spreads[order],curve[order]))
        value=mean(sampler,m);base=mean(curve_configs[0],m)
        res[m]={'value':value,'effect':value-base,'residual':value-at}
    return res


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--probe',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--reference',nargs='+',required=True,help='tau=1 accurate config per probe, e.g. heun32_t1.0')
    p.add_argument('--range',type=int,nargs=2,default=[100,160]);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    keep={f'context_{i:03d}' for i in range(*args.range)};rng=np.random.default_rng(20261026);result={'margins':MARGINS,'range':args.range,'models':{}}
    for probe,label,reference in zip(args.probe,args.labels,args.reference):
        seeds=json.loads((probe/'manifest.json').read_text())['seeds']
        with np.load(probe/'plans.npz') as z:plans={k:z[k] for k in z.files}
        names=sorted(n for n in {k.split('_s')[0] for k in plans if '_s92' in k} if n in keep)
        configs=sorted({k.split('_',3)[-1] for k in plans if '_s92' in k})
        human={n:plans[f'{n}_human'] if f'{n}_human' in plans else np.load(args.data/f'{n}.npz')['ego_future_xyz'] for n in names}
        method=reference.split('_t')[0];curve=sorted([c for c in configs if c.startswith(method+'_t')],key=lambda c:-float(c.split('_t')[1]))
        taus=[float(c.split('_t')[1]) for c in curve];samplers=[c for c in configs if c.startswith('euler') and c.endswith('_t1.0') and not c.startswith(method)]
        table=per_clip(plans,names,configs,seeds,reference,human,{n:records[n]['speed_mps'] for n in names})
        model={'clips':len(names),'seeds':len(seeds),'reference':reference,'samplers':{}}
        for sampler in samplers:
            point=summarize(table,names,curve,taus,sampler)
            boots=[summarize(table,list(rng.choice(names,len(names))),curve,taus,sampler) for _ in range(args.boot)]
            entry={'tau_eff':[point['tau_eff'],*np.quantile([b['tau_eff'] for b in boots],[.025,.975]).tolist()]}
            for m in METRICS:
                eff=[b[m]['effect'] for b in boots];resid=[b[m]['residual'] for b in boots]
                lo,hi=np.quantile(resid,[.05,.95])  # 90% interval for TOST at alpha=0.05
                frac=[1-r/e if abs(e)>1e-9 else np.nan for r,e in zip(resid,eff)]
                entry[m]={'value':point[m]['value'],'effect':[point[m]['effect'],*np.quantile(eff,[.025,.975]).tolist()],
                    'residual':[point[m]['residual'],*np.quantile(resid,[.025,.975]).tolist()],
                    'equivalent_within_margin':bool(-MARGINS[m]<lo and hi<MARGINS[m]),
                    'fraction_explained':float(1-point[m]['residual']/point[m]['effect']) if abs(point[m]['effect'])>1e-9 else None,
                    'fraction_explained_ci':np.nanquantile(frac,[.025,.975]).tolist()}
            model['samplers'][sampler]=entry
        result['models'][label]=model
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for label,model in result['models'].items():
        print(f"== {label} ({model['clips']} clips, ref {model['reference']})")
        for s,e in model['samplers'].items():
            line=f"  {s:13s} tau_eff {e['tau_eff'][0]:.2f} [{e['tau_eff'][1]:.2f},{e['tau_eff'][2]:.2f}]"
            for m in ['ade1','minade6','flip','stop','decelerate','entropy']:
                r=e[m];line+=f" | {m} eff {r['effect'][0]:+.3f} res {r['residual'][0]:+.3f}[{r['residual'][1]:+.3f},{r['residual'][2]:+.3f}]{'≡' if r['equivalent_within_margin'] else ''}"
            print(line)


if __name__=='__main__':main()
