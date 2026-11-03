"""Addendum 20 (pre-registered 30/10/2026 09:10 PDT): E10 against the converged reference on Alpamayo-R1 and Alpamayo-2.

Per model, held-out pooled400 clips context_000-399, seeds 9201-9206, same initial noise. Curves:
  heun64 : Heun-64 at tau in {1,.9,.85,.8,.75,.7} (the A20 run, all 400 clips)
  heun32 : the principal Heun-32 grid tau in {1,.9,.8,.7,.6,.5,.4} (the model's pooled400 probe; the paper's curve)
Samplers:
  euler10_a20      : E10 tau=1 from the A20 run (same hardware as the Heun-64 reference) -- PRIMARY
  euler10_pooled400: E10 tau=1 from the model's pooled400 probe -- SENSITIVITY (hardware effect)
Machinery is the addendum-19 analysis (convergence_a19_analysis): per-clip metrics from paper_numbers.clip_metrics, flips
against the tau=1 reference of the SAME solver as the curve, tau_eff by spread matching and linear interpolation
(convergence_a18_analysis.matched), TOST at alpha=.05 from the 90% bootstrap interval, margins paper_numbers.MARGINS
(ADE1 0.05 m, flips 2 pt).

Registered predictions (per model):
 F1 E10 flip residual against heun64 equivalent (TOST 90% within +-2 pt).
 F2 E10 ADE1 residual against heun64 equivalent (+-0.05 m).
 F3 |tau_eff(E10 vs heun64) - tau_eff(E10 vs heun32)| < 0.02 on the same 400 clips (point estimate, as in A19; CIs reported).
    Reading (closest to A19): the same E10 against both curves; primary = A20 E10, sensitivity = pooled400 E10.
    The difference to the paper's tau_eff (pooled400 E10 vs heun32) is also reported, unregistered.
Clip bootstrap: 2000 resamples, seed 20261101, the same resamples for every curve and sampler (joint), re-solving tau_eff.
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
REGISTERED=['heun64','heun32'];PRIMARY='euler10_a20';SENSITIVITY='euler10_pooled400'
SAMPLERS=[PRIMARY,SENSITIVITY];METRICS=['spread','ade1','flip']
DEFAULTS={'r1':('artifacts/temperature_probe_r1_heldout_pooled400_v3_20261028','artifacts/convergence_a20_r1_heun64_heldout_v3_20261030'),
    'a2':('artifacts/temperature_probe_a2_heldout_pooled400_v3_20261028','artifacts/convergence_a20_a2_heun64_heldout_v3_20261030')}
cfg=lambda solver,t:f'{solver}_t{t}'
ci90=lambda x:[float(x[0]),*np.quantile(x[1:],[.05,.95]).tolist()]


def load(args):
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*args.range)];plans={}
    probe=json.loads((args.probe/'manifest.json').read_text());run=json.loads((args.a20/'manifest.json').read_text())
    seeds=probe['seeds'];assert run['seeds']==seeds,(run['seeds'],seeds)
    for m in (probe,run):
        assert Path(m['data'])==args.data,(m['data'],args.data);assert m['model']==args.model,(m['model'],args.model)
        assert m['range'][0]<=args.range[0] and args.range[1]<=m['range'][1],m['range']
    with np.load(args.probe/'plans.npz') as z:
        for k in z.files:
            if k.split('_s')[0] in names:plans[k.replace('_euler10_t1.0','_euler10_pooled400') if k.endswith('_euler10_t1.0') else k]=z[k]
    with np.load(args.a20/'plans.npz') as z:  # Heun-64 grid + E10 tau=1 (renamed, the probe has its own E10)
        new={(k.replace('_euler10_t1.0','_euler10_a20') if k.endswith('_euler10_t1.0') else k):z[k]
             for k in z.files if k.split('_s')[0] in names and ('_heun64_t' in k or k.endswith('_euler10_t1.0'))}
    assert not set(new)&set(plans);plans.update(new)
    for n in names:
        for s in seeds:
            for c in SAMPLERS+['euler1_t1.0','heun32_t1.0']+[cfg(sv,t) for sv,ts in CURVES.values() for t in ts]:
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
    # how different the two E10 runs are (same seeds and initial noise; differences = hardware/numerics)
    diff=np.array([max(np.abs(plans[f'{n}_s{s}_{PRIMARY}']-plans[f'{n}_s{s}_{SENSITIVITY}']).max() for s in seeds) for n in names])
    e10={'max_abs_plan_difference_m':float(diff.max()),'clips_bit_identical':int((diff==0).sum()),
         'decision_agreement':float(np.mean([decisions(plans[f'{n}_s{s}_{PRIMARY}'],records[n]['speed_mps'])==
                                             decisions(plans[f'{n}_s{s}_{SENSITIVITY}'],records[n]['speed_mps']) for n in names for s in seeds]))}
    return names,seeds,tables,e10


def entry(r):
    e={'tau_eff':ci(r['tau_eff']),'tau_eff_ci90':ci90(r['tau_eff']),'in_range':bool(r['in_range'][0])}
    for m in ('ade1','flip'):
        lo,hi=np.quantile(r[m]['residual'][1:],[.05,.95])
        e[m]={'reference':float(r[m]['reference'][0]),'sampler':float(r[m]['sampler'][0]),'matched':float(r[m]['matched'][0]),
            'effect':ci(r[m]['effect']),'residual':ci(r[m]['residual']),'tost90':[float(lo),float(hi)],
            'equivalent':bool(-MARGINS[m]<lo and hi<MARGINS[m])}
    return e


def tests(runs,match,a,b,s):
    e=match[a][s];d=runs[a][s]['tau_eff']-runs[b][s]['tau_eff']
    return {'sampler':s,
        'F1':{'curve':a,'flip_residual_pt':100*e['flip']['residual'][0],'tost90_pt':[100*x for x in e['flip']['tost90']],
            'margin_pt':100*MARGINS['flip'],'holds':e['flip']['equivalent']},
        'F2':{'curve':a,'ade1_residual_m':e['ade1']['residual'][0],'tost90_m':e['ade1']['tost90'],'margin_m':MARGINS['ade1'],
            'holds':e['ade1']['equivalent']},
        'F3':{'curves':[a,b],'tau_eff':{a:e['tau_eff'],b:match[b][s]['tau_eff']},'difference':ci(d),'difference_ci90':ci90(d),
            'abs_difference':float(abs(d[0])),'margin':.02,'holds':bool(abs(d[0])<.02)}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=sorted(DEFAULTS),required=True)
    p.add_argument('--data',type=Path,default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--probe',type=Path,default=None,help='pooled400 probe (Heun-32 grid, pooled400 E10)')
    p.add_argument('--a20',type=Path,default=None,help='merged A20 run (Heun-64 grid, E10)')
    p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261101);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.probe=args.probe or Path(DEFAULTS[args.model][0]);args.a20=args.a20 or Path(DEFAULTS[args.model][1])
    names,seeds,tables,e10=load(args);rng=np.random.default_rng(args.seed)
    idx=np.vstack([np.arange(len(names))]+[rng.choice(len(names),len(names)) for _ in range(args.boot)])
    distinct=[int(len(np.unique(i))) for i in idx[1:]]
    result={'addendum':20,'model':args.model,'clips':len(names),'distinct_clips':len(set(names)),'range':args.range,'seeds':seeds,
        'boot':args.boot,'boot_seed':args.seed,'boot_distinct_clips_per_resample':[float(np.mean(distinct)),min(distinct),max(distinct)],
        'margins':MARGINS,'inputs':{'probe':str(args.probe),'a20':str(args.a20),'data':str(args.data)},
        'samplers':{PRIMARY:'E10 tau=1 from the A20 run (primary)',SENSITIVITY:'E10 tau=1 from the pooled400 probe (sensitivity)'},
        'e10_run_vs_pooled400':e10,
        'curves':{k:{'solver':s,'taus':t,'registered':k in REGISTERED} for k,(s,t) in CURVES.items()}}
    runs={k:{s:matched(tables[solver],idx,s,[cfg(solver,t) for t in taus],taus) for s in SAMPLERS} for k,(solver,taus) in CURVES.items()}
    match={k:{s:entry(r) for s,r in per.items()} for k,per in runs.items()}
    result['matched']=match
    result['primary']=tests(runs,match,'heun64','heun32',PRIMARY)
    result['sensitivity']=tests(runs,match,'heun64','heun32',SENSITIVITY)
    dp=runs['heun64'][PRIMARY]['tau_eff']-runs['heun32'][SENSITIVITY]['tau_eff']
    result['unregistered_tau_eff_vs_paper']={'note':'tau_eff(A20 E10 vs heun64) - tau_eff(pooled400 E10 vs heun32, the paper value)',
        'difference':ci(dp),'difference_ci90':ci90(dp)}
    ex={}
    for s in SAMPLERS:
        t=tests(runs,match,'heun64_step01','heun32',s);t['F3_same_grid']=tests(runs,match,'heun64_step01','heun32_step01',s)['F3'];ex[s]=t
    ex['note']='exploratory: Heun-64 curve restricted to tau in {1,.9,.8,.7}; not a registered test'
    result['exploratory_grid']=ex
    result['reference_tau1']={k:{m:{'mean':float(tables[k][cfg(k,1.0)][m].mean())} for m in ('spread','ade1')} for k in ('heun64','heun32')}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'results.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(f"== Addendum 20, model {args.model}, {len(names)} clips, {args.boot} resamples seed {args.seed}; "
          f"distinct clips/resample {result['boot_distinct_clips_per_resample']}")
    print('  E10 A20 vs pooled400:',e10)
    for k in CURVES:
        for s in SAMPLERS:
            e=match[k][s]
            print(f"  {k:14s} {s:18s} tau {e['tau_eff'][0]:.3f} [{e['tau_eff'][1]:.3f},{e['tau_eff'][2]:.3f}]{'' if e['in_range'] else ' OUT'}"
                  f" | ade1 res {e['ade1']['residual'][0]:+.4f} tost {e['ade1']['tost90'][0]:+.4f},{e['ade1']['tost90'][1]:+.4f} eq {e['ade1']['equivalent']}"
                  f" | flip res {100*e['flip']['residual'][0]:+.2f} tost {100*e['flip']['tost90'][0]:+.2f},{100*e['flip']['tost90'][1]:+.2f} eq {e['flip']['equivalent']}")
    for part in ('primary','sensitivity'):
        for f in ('F1','F2','F3'):print(' ',part,f,result[part][f])
    print('  vs paper tau_eff',result['unregistered_tau_eff_vs_paper'])
    print('  exploratory',json.dumps(ex))


if __name__=='__main__':main()
