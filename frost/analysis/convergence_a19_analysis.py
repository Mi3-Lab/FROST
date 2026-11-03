"""Addendum 19 (pre-registered 30/10/2026 00:33 PDT): E10 flip equivalence against the converged reference on 400 clips.

Alpamayo-1.5, held-out pooled400 clips context_000-399, seeds 9201-9206, same initial noise. Curves:
  heun64 : Heun-64 at tau in {1,.9,.85,.8,.75,.7} (clips 0-99 from addendum 18, 100-399 from the A19 run)
  heun32 : the principal Heun-32 grid tau in {1,.9,.8,.7,.6,.5,.4} (pooled400 probe; the paper's curve)
E10 and E4 come from the pooled400 probe. Flips are counted against the tau=1 reference of the SAME solver as the
curve (same seed). Per-clip metrics from frost.analysis.paper_numbers.clip_metrics; tau_eff / residuals use the spread
matching and linear interpolation of convergence_a18_analysis.matched (= paper_numbers.matched, parametrized by
curve); TOST at alpha=.05 from the 90% bootstrap interval of the residual, margins paper_numbers.MARGINS
(ADE1 0.05 m, flips 2 pt).

Registered predictions:
 D1 E10 flip residual against heun64 equivalent (TOST 90% within +-2 pt).
 D2 E10 ADE1 residual against heun64 equivalent (+-0.05 m).
 D3 |tau_eff(E10 vs heun64) - tau_eff(E10 vs heun32)| < 0.02 on the same 400 clips (point estimate; CI reported).
Clip bootstrap: 2000 resamples, seed 20261031, the same resamples for every curve (joint), re-solving tau_eff.
Exploratory: heun64 restricted to tau in {1,.9,.8,.7} (grid effect); heun32 restricted to the same grid.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.analysis.paper_numbers import clip_metrics, MARGINS
from frost.analysis.convergence_a18_analysis import matched, ci

GRID=[1.,.9,.85,.8,.75,.7];PRINCIPAL=[1.,.9,.8,.7,.6,.5,.4];STEP01=[1.,.9,.8,.7]
CURVES={'heun64':('heun64',GRID),'heun32':('heun32',PRINCIPAL),
    # exploratory: grid effect
    'heun64_step01':('heun64',STEP01),'heun32_step01':('heun32',STEP01)}
REGISTERED=['heun64','heun32'];SAMPLERS=['euler10_t1.0','euler4_t1.0'];METRICS=['spread','ade1','flip']
cfg=lambda solver,t:f'{solver}_t{t}'


def load(args):
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*args.range)];plans={}
    seeds=json.loads((args.probe/'manifest.json').read_text())['seeds']
    with np.load(args.probe/'plans.npz') as z:plans.update({k:z[k] for k in z.files if k.split('_s')[0] in names})
    for path in (args.a18,args.a19):  # Heun-64 only; A18 also holds Heun-128 and dense Heun-32 (not used here)
        assert json.loads((path/'manifest.json').read_text())['seeds']==seeds
        with np.load(path/'plans.npz') as z:
            new={k:z[k] for k in z.files if k.split('_s')[0] in names and '_heun64_t' in k}
        assert not set(new)&set(plans);plans.update(new)
    for n in names:
        for s in seeds:
            for c in SAMPLERS+['euler1_t1.0']+[cfg(sv,t) for sv,ts in CURVES.values() for t in ts]:
                assert f'{n}_s{s}_{c}' in plans,(n,s,c)
    tables={}  # tables[solver][config] -> {metric: array over clips}, flips against solver's own tau=1
    for solver in ('heun64','heun32'):
        configs=SAMPLERS+sorted({cfg(sv,t) for sv,ts in CURVES.values() if sv==solver for t in ts})
        rows={c:{m:[] for m in METRICS} for c in configs}
        for n in names:
            human=np.load(args.data/f'{n}.npz')['ego_future_xyz'];speed=records[n]['speed_mps']
            ref_dec=[decisions(plans[f'{n}_s{s}_{solver}_t1.0'],speed) for s in seeds]
            for c in configs:
                r=clip_metrics(plans,n,c,seeds,speed,human,ref_dec)
                for m in METRICS:rows[c][m].append(r[m])
        tables[solver]={c:{m:np.array(v) for m,v in d.items()} for c,d in rows.items()}
    return names,seeds,tables


def entry(r):
    e={'tau_eff':ci(r['tau_eff']),'in_range':bool(r['in_range'][0])}
    for m in ('ade1','flip'):
        lo,hi=np.quantile(r[m]['residual'][1:],[.05,.95])
        e[m]={'reference':float(r[m]['reference'][0]),'sampler':float(r[m]['sampler'][0]),'matched':float(r[m]['matched'][0]),
            'effect':ci(r[m]['effect']),'residual':ci(r[m]['residual']),'tost90':[float(lo),float(hi)],
            'equivalent':bool(-MARGINS[m]<lo and hi<MARGINS[m])}
    return e


def tests(runs,match,a,b):
    e=match[a]['euler10_t1.0'];d=runs[a]['euler10_t1.0']['tau_eff']-runs[b]['euler10_t1.0']['tau_eff']
    return {'D1':{'curve':a,'flip_residual_pt':100*e['flip']['residual'][0],'tost90_pt':[100*x for x in e['flip']['tost90']],
            'margin_pt':100*MARGINS['flip'],'holds':e['flip']['equivalent']},
        'D2':{'curve':a,'ade1_residual_m':e['ade1']['residual'][0],'tost90_m':e['ade1']['tost90'],'margin_m':MARGINS['ade1'],
            'holds':e['ade1']['equivalent']},
        'D3':{'curves':[a,b],'tau_eff':{a:e['tau_eff'],b:match[b]['euler10_t1.0']['tau_eff']},'difference':ci(d),
            'abs_difference':float(abs(d[0])),'margin':.02,'holds':bool(abs(d[0])<.02)}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--probe',type=Path,default=Path('artifacts/temperature_probe_a15_heldout_pooled400_v3_20261028'))
    p.add_argument('--a18',type=Path,default=Path('artifacts/convergence_a18_a15_heldout_v3_20261029'))
    p.add_argument('--a19',type=Path,default=Path('artifacts/convergence_a19_a15_heun64_heldout_v3_20261030'))
    p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261031);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    names,seeds,tables=load(args);rng=np.random.default_rng(args.seed)
    idx=np.vstack([np.arange(len(names))]+[rng.choice(len(names),len(names)) for _ in range(args.boot)])
    distinct=[int(len(np.unique(i))) for i in idx[1:]]
    result={'addendum':19,'clips':len(names),'distinct_clips':len(set(names)),'range':args.range,'seeds':seeds,'boot':args.boot,
        'boot_seed':args.seed,'boot_distinct_clips_per_resample':[float(np.mean(distinct)),min(distinct),max(distinct)],'margins':MARGINS,
        'inputs':{'probe':str(args.probe),'heun64_clips_000_099':str(args.a18),'heun64_clips_100_399':str(args.a19),'data':str(args.data)},
        'curves':{k:{'solver':s,'taus':t,'registered':k in REGISTERED} for k,(s,t) in CURVES.items()}}
    runs={k:{s:matched(tables[solver],idx,s,[cfg(solver,t) for t in taus],taus) for s in SAMPLERS} for k,(solver,taus) in CURVES.items()}
    match={k:{s:entry(r) for s,r in per.items()} for k,per in runs.items()}
    result['matched']=match
    result.update(tests(runs,match,'heun64','heun32'))
    ex=tests(runs,match,'heun64_step01','heun32')
    ex['D3_same_grid']=tests(runs,match,'heun64_step01','heun32_step01')['D3']
    ex['note']='exploratory: Heun-64 curve restricted to tau in {1,.9,.8,.7}; not a registered test'
    result['exploratory_grid']=ex
    result['reference_tau1']={k:{m:{'mean':float(tables[k][cfg(k,1.0)][m].mean())} for m in ('spread','ade1')} for k in ('heun64','heun32')}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'results.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(f"== Addendum 19, {len(names)} clips, {args.boot} resamples seed {args.seed}; distinct clips/resample {result['boot_distinct_clips_per_resample']}")
    for k in CURVES:
        for s in SAMPLERS:
            e=match[k][s]
            print(f"  {k:14s} {s:12s} tau {e['tau_eff'][0]:.3f} [{e['tau_eff'][1]:.3f},{e['tau_eff'][2]:.3f}]{'' if e['in_range'] else ' OUT'}"
                  f" | ade1 res {e['ade1']['residual'][0]:+.4f} tost {e['ade1']['tost90'][0]:+.4f},{e['ade1']['tost90'][1]:+.4f} eq {e['ade1']['equivalent']}"
                  f" | flip res {100*e['flip']['residual'][0]:+.2f} tost {100*e['flip']['tost90'][0]:+.2f},{100*e['flip']['tost90'][1]:+.2f} eq {e['flip']['equivalent']}")
    for d in ('D1','D2','D3'):print(' ',d,result[d])
    print('  exploratory',json.dumps(ex))


if __name__=='__main__':main()
