"""Exploratory decision flips and minADE6 for the FlashDrive probes (protocols/frost_main.md, addendum 11c).

Per window and seed, flips are decision changes against Heun32 tau=1 at the same seed and window
(trajectory_decision_gate.decisions, current speed from the clip record); minADE6 is the best of the six
seeds against the window's human future. Clip means; tau_eff from seed spread as in flashdrive_analysis;
residual against the Heun32 tau curve at tau_eff; 1000-draw clip bootstrap re-solving tau_eff.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions

TAUS=[.5,.6,.7,.8,.9,1.];SAMPLERS=['flashcache8_t1.0','euler4_t1.0','euler8_t1.0','euler10_t1.0'];REFERENCE='heun32_t1.0'


def per_clip(path,speeds):
    z=np.load(path);windows=defaultdict(dict)
    for k in z.files:
        clip,rest=k.split('_w');w,_,tail=rest.partition('_');windows[(clip,w)][tail]=z[k]
    clips=defaultdict(lambda:defaultdict(list));key=lambda x:(x['longitudinal'],x['lateral'])
    for (clip,w),d in windows.items():
        if 'human' not in d or not any(t.startswith('s') for t in d):continue
        human=d['human'][:,:2];seeds=sorted({t.split('_',1)[0] for t in d if t.startswith('s')})
        ref={s:key(decisions(d[f'{s}_{REFERENCE}'],speeds[clip])) for s in seeds}
        for cfg in {t.split('_',1)[1] for t in d if t.startswith('s')}:
            trajs={s:d[f'{s}_{cfg}'] for s in seeds}
            spread=np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs.values()) for b in list(trajs.values())[i+1:]])
            flip=np.mean([key(decisions(trajs[s],speeds[clip]))!=ref[s] for s in seeds])
            minade=min(np.linalg.norm(t[:,:2]-human,axis=1).mean() for t in trajs.values())
            clips[clip][cfg].append((spread,flip,minade))
    return {c:{cfg:np.mean(v,axis=0) for cfg,v in d.items()} for c,d in clips.items()}


def matched(table,clips,sampler):
    curve=np.array([np.mean([table[c][f'heun32_t{t}'] for c in clips],axis=0) for t in TAUS])
    spread,flip,minade=np.mean([table[c][sampler] for c in clips],axis=0)
    if not curve[0,0]<=spread<=curve[-1,0]:return None
    tau=float(np.interp(spread,curve[:,0],TAUS))
    at=lambda j:float(np.interp(tau,TAUS,curve[:,j]))
    return {'tau_eff':tau,'flip':float(flip),'flip_matched':at(1),'flip_resid':float(flip)-at(1),
        'minade6':float(minade),'minade6_ref':float(curve[-1,2]),'minade6_matched':at(2),'minade6_resid':float(minade)-at(2)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,nargs='+',required=True)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--boot',type=int,default=1000)
    args=p.parse_args();speeds={r['name']:r['speed_mps'] for r in json.loads(args.records.read_text()) if 'error' not in r};result={}
    for probe in args.probe:
        table=per_clip(probe/'plans.npz',speeds);clips=sorted(table);rng=np.random.default_rng(20261026)
        draws=[rng.choice(clips,len(clips)) for _ in range(args.boot)];entry={'clips':len(clips),'samplers':{}}
        for s in SAMPLERS:
            m=matched(table,clips,s)
            if m:
                b=[x for x in (matched(table,d,s) for d in draws) if x]
                for f in ['flip_resid','minade6_resid','tau_eff']:m[f+'_ci90']=np.quantile([x[f] for x in b],[.05,.95]).tolist()
                m['boot_in_range']=len(b)/args.boot
            entry['samplers'][s]=m
        result[probe.name]=entry
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for pr,e in result.items():
        for s,m in e['samplers'].items():
            if m:print(pr,s,f"tau {m['tau_eff']:.2f} flip {100*m['flip']:.1f} vs {100*m['flip_matched']:.1f} resid {100*m['flip_resid']:+.1f} {[round(100*x,1) for x in m['flip_resid_ci90']]} | minADE6 {m['minade6']:.3f} ref {m['minade6_ref']:.3f} resid {m['minade6_resid']:+.3f} {[round(x,3) for x in m['minade6_resid_ci90']]}")


if __name__=='__main__':main()
