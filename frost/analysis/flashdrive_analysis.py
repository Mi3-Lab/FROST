"""Matched-temperature analysis of the FlashDrive probe (protocols/frost_main.md, addenda 11/11b).

Clip is the unit (mean over windows). For each sampler, tau_eff is where the Heun32 curve's mean
seed spread equals the sampler's; the ADE1 residual is sampler minus the curve at tau_eff, with a
clip bootstrap that re-solves tau_eff in every resample.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

TAUS=[.5,.6,.7,.8,.9,1.];SAMPLERS=['flashcache8_t1.0','euler4_t1.0','euler8_t1.0','euler10_t1.0']


def per_clip(path):
    z=np.load(path);keys=defaultdict(dict)
    for k in z.files:
        clip,rest=k.split('_w');w,_,tail=rest.partition('_');keys[(clip,w)][tail]=z[k]
    clips=defaultdict(lambda:defaultdict(list))
    for (clip,w),d in keys.items():
        if 'human' not in d or not any(t.startswith('s') for t in d):continue
        cfgs=defaultdict(list)
        for t,v in d.items():
            if t.startswith('s'):cfgs[t.split('_',1)[1]].append(v[:,:2])
        for cfg,trajs in cfgs.items():
            ade=np.mean([np.linalg.norm(a-d['human'][:,:2],axis=1).mean() for a in trajs])
            spread=np.mean([np.linalg.norm(a-b,axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]])
            clips[clip][cfg].append((ade,spread))
    return {c:{cfg:np.mean(v,axis=0) for cfg,v in d.items()} for c,d in clips.items()}


def matched(table,clips,sampler):
    curve=np.array([np.mean([table[c][f'heun32_t{t}'] for c in clips],axis=0) for t in TAUS])
    ade,spread=np.mean([table[c][sampler] for c in clips],axis=0)
    if not curve[0,1]<=spread<=curve[-1,1]:return None
    tau=float(np.interp(spread,curve[:,1],TAUS));ref=float(np.interp(tau,TAUS,curve[:,0]))
    return tau,float(ade),ref,float(ade-ref)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--boot',type=int,default=1000);args=p.parse_args()
    table=per_clip(args.probe/'plans.npz');clips=sorted(table);rng=np.random.default_rng(20261026)
    result={'clips':len(clips),'curve':{t:np.mean([table[c][f'heun32_t{t}'] for c in clips],axis=0).tolist() for t in TAUS},'samplers':{}}
    draws=[rng.choice(clips,len(clips)) for _ in range(args.boot)]
    for s in SAMPLERS:
        m=matched(table,clips,s);ade,spread=np.mean([table[c][s] for c in clips],axis=0)
        entry={'ade1':float(ade),'spread':float(spread),'in_range':m is not None}
        if m:
            b=[x for x in (matched(table,d,s) for d in draws) if x]
            entry.update(tau_eff=m[0],matched_ade1=m[2],residual=m[3],residual_ci90=np.quantile([x[3] for x in b],[.05,.95]).tolist(),
                tau_ci90=np.quantile([x[0] for x in b],[.05,.95]).tolist(),boot_in_range=len(b)/args.boot)
        result['samplers'][s]=entry
    # Paired per-clip ADE1 differences between samplers from the same reasoning and noise (exploratory, added after review).
    pairs=[('flashcache8_t1.0','euler8_t1.0'),('flashcache8_t1.0','euler10_t1.0'),('flashcache8_t1.0','euler4_t1.0'),('euler10_t1.0','heun32_t1.0')]
    result['paired_ade1']={}
    for a,b in pairs:
        d=np.array([table[c][a][0]-table[c][b][0] for c in clips]);idx={c:i for i,c in enumerate(clips)}
        boot=[d[[idx[c] for c in draw]].mean() for draw in draws]
        result['paired_ade1'][f'{a}-{b}']={'mean':float(d.mean()),'ci90':np.quantile(boot,[.05,.95]).tolist()}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=1))


if __name__=='__main__':main()
