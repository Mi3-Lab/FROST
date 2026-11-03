"""Summarize explicit vs implicit temperature: single-sample error, coverage and decision distribution per NFE."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions


def entropy(labels):
    c=np.array(list(collections.Counter(labels).values()),float)/len(labels);return float(-(c*np.log2(c)).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',default='heun32_t1.0',help='Config treated as the converged tau=1 sampler');args=p.parse_args()
    manifest=json.loads((args.probe/'manifest.json').read_text());seeds=manifest['seeds']
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    with np.load(args.probe/'plans.npz') as z:plans={k:z[k] for k in z.files}
    names=sorted({k.split('_s')[0] for k in plans if '_s92' in k});configs=sorted({k.split('_',3)[-1] for k in plans if '_s92' in k})
    rng=np.random.default_rng(0);rows=[]
    # Probes that build their own scenes (Qwen-Drive) store the matching logged future in the plans file.
    human={n:plans[f'{n}_human'] if f'{n}_human' in plans else np.load(args.data/f'{n}.npz')['ego_future_xyz'] for n in names}
    hclass={n:decisions(human[n],records[n]['speed_mps'])['longitudinal'] for n in names}
    for cfg in configs:
        method,rest=cfg.split('_t');tau=float(rest);steps=int(''.join(ch for ch in method if ch.isdigit()));nfe=steps*(2 if method.startswith('heun') else 1)
        ade1=[];minade=[];spread=[];ent=[];modal=[];stop=[];decel=[];flip=[]
        for n in names:
            trajs=[plans[f'{n}_s{s}_{cfg}'] for s in seeds];sp=records[n]['speed_mps']
            e=[np.linalg.norm(t[:,:2]-human[n][:,:2],axis=1).mean() for t in trajs]
            ade1.append(np.mean(e));minade.append(min(e))
            spread.append(np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]]))
            d=[decisions(t,sp) for t in trajs];lon=[x['longitudinal'] for x in d];full=[(x['longitudinal'],x['lateral']) for x in d]
            ent.append(entropy(full));modal.append(collections.Counter(lon).most_common(1)[0][0]==hclass[n])
            stop.append(np.mean([x=='stop' for x in lon]));decel.append(np.mean([x=='decelerate' for x in lon]))
            ref=[decisions(plans[f'{n}_s{s}_{args.reference}'],sp) for s in seeds]
            flip.append(np.mean([(a['longitudinal'],a['lateral'])!=(b['longitudinal'],b['lateral']) for a,b in zip(d,ref)]))
        def ci(x):
            x=np.array(x,float);b=[x[rng.integers(len(x),size=len(x))].mean() for _ in range(1000)];return [float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
        rows.append({'config':cfg,'method':method.rstrip('0123456789'),'steps':steps,'tau':tau,'nfe':nfe,'ade1':ci(ade1),'minade6':ci(minade),
            'spread':ci(spread),'decision_entropy':ci(ent),'modal_agrees_human':ci(modal),'stop':ci(stop),'decelerate':ci(decel),'flip_vs_reference':ci(flip)})
    rows.sort(key=lambda r:(r['nfe'],r['method'],-r['tau']))
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'clips':len(names),'seeds':seeds,'rows':rows},indent=2))
    print(f"{'config':18s} nfe  ADE1   minADE6 spread  H_dec  modal  stop   decel  flip")
    for r in rows:print(f"{r['config']:18s} {r['nfe']:3d}  {r['ade1'][0]:.3f}  {r['minade6'][0]:.3f}   {r['spread'][0]:.3f}  {r['decision_entropy'][0]:.3f}  {r['modal_agrees_human'][0]:.3f}  {r['stop'][0]:.3f}  {r['decelerate'][0]:.3f}  {r['flip_vs_reference'][0]:.3f}")


if __name__=='__main__':main()
