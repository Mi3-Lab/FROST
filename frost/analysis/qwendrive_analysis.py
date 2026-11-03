"""Matched-temperature analysis of the Qwen-Drive v2 probe (protocols/frost_main.md, addendum 12).

Clip is the unit. For each release sampler (Euler 1/2/4/10, ending on the endpoint at t = 0.9), tau_eff is where
the mean seed spread of the Heun-32 curve on [0, 0.9] equals the sampler's (linear interpolation on the tau grid);
ADE1, minADE6 and decision-flip residuals are sampler minus the curve at tau_eff, with a clip bootstrap that
re-solves tau_eff in every resample. Flips are against the reference's decision for the same seed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions

TAUS=[1.,.9,.8,.7,.6,.5,.4,.3];CURVE=[f'heun32_t{t}' for t in TAUS];REFERENCE='heun32_t1.0'
SAMPLERS=[f'euler{k}_t1.0' for k in [1,2,4,10]];METRICS=['ade1','minade6','flip'];MARGINS={'ade1':.05,'flip':.02}


def clip_table(plans,records):
    names=sorted(k.rsplit('_human',1)[0] for k in plans if k.endswith('_human'))
    seeds=sorted({int(k.split('_s')[1].split('_')[0]) for k in plans if '_s' in k and not k.endswith('_human')})
    configs=sorted({k.split('_',3)[3] for k in plans if not k.endswith('_human')});table={}
    key=lambda x:(x['longitudinal'],x['lateral'])
    for n in names:
        human=plans[f'{n}_human'][:,:2];speed=records[n]['speed_mps'];ref=[key(decisions(plans[f'{n}_s{s}_{REFERENCE}'],speed)) for s in seeds]
        table[n]={}
        for c in configs:
            trajs=[plans[f'{n}_s{s}_{c}'][:,:2] for s in seeds];err=[np.linalg.norm(t-human,axis=1).mean() for t in trajs]
            table[n][c]={'spread':float(np.mean([np.linalg.norm(a-b,axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]])),
                'ade1':float(np.mean(err)),'minade6':float(min(err)),
                'flip':float(np.mean([key(decisions(plans[f'{n}_s{s}_{c}'],speed))!=r for s,r in zip(seeds,ref)]))}
    return names,seeds,table


def matched(table,sample,sampler):
    mean=lambda c,m:float(np.mean([table[n][c][m] for n in sample]))
    spreads=np.array([mean(c,'spread') for c in CURVE]);o=np.argsort(spreads);s=mean(sampler,'spread')
    out={'tau_eff':float(np.interp(s,spreads[o],np.array(TAUS)[o])),'in_range':bool(spreads.min()<=s<=spreads.max()),'spread':s}
    for m in METRICS:
        at=float(np.interp(s,spreads[o],np.array([mean(c,m) for c in CURVE])[o]));v=mean(sampler,m)
        out[m]={'reference':mean(REFERENCE,m),'sampler':v,'matched':at,'effect':v-mean(REFERENCE,m),'residual':v-at}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,nargs='+',required=True)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--boot',type=int,default=1000)
    args=p.parse_args();records={r['name']:r for r in json.loads(args.records.read_text()) if 'error' not in r};result={'margins':MARGINS,'planners':{}}
    for probe in args.probe:
        with np.load(probe/'plans.npz') as z:plans={k:z[k] for k in z.files}
        names,seeds,table=clip_table(plans,records);rng=np.random.default_rng(20261027);draws=[list(rng.choice(names,len(names))) for _ in range(args.boot)]
        entry={'clips':len(names),'seeds':seeds,'curve_spread':{c:float(np.mean([table[n][c]['spread'] for n in names])) for c in CURVE},
            'heun64_vs_heun32_max_m':float(max(np.abs(plans[f'{n}_s{s}_heun64_t1.0']-plans[f'{n}_s{s}_{REFERENCE}'])[:,:2].max() for n in names for s in seeds)),'samplers':{}}
        for s in SAMPLERS:
            point=matched(table,names,s);boot=[matched(table,d,s) for d in draws]
            point['tau_eff_ci90']=np.quantile([b['tau_eff'] for b in boot],[.05,.95]).tolist();point['boot_in_range']=float(np.mean([b['in_range'] for b in boot]))
            for m in METRICS:
                for f in ['effect','residual']:point[m][f+'_ci90']=np.quantile([b[m][f] for b in boot],[.05,.95]).tolist()
                if m in MARGINS:lo,hi=point[m]['residual_ci90'];point[m]['equivalent']=bool(-MARGINS[m]<lo and hi<MARGINS[m])
            entry['samplers'][s]=point
        result['planners'][probe.name]=entry
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for pl,e in result['planners'].items():
        print(pl,e['clips'],'heun64 diff',round(e['heun64_vs_heun32_max_m'],4),'curve',{k:round(v,3) for k,v in e['curve_spread'].items()})
        for s,v in e['samplers'].items():
            print(' ',s,'tau',round(v['tau_eff'],2),[round(x,2) for x in v['tau_eff_ci90']],'inrange',v['in_range'],v['boot_in_range'],
                *[f"{m} eff {v[m]['effect']:+.3f} res {v[m]['residual']:+.3f} {[round(x,3) for x in v[m]['residual_ci90']]} eq={v[m].get('equivalent')}" for m in METRICS])


if __name__=='__main__':main()
