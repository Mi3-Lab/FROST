"""Thor v1 analysis (protocols/thor.md sections 4-6; registered 29/10/2026 14:59 PDT), run on the workstation
on a copy of the board's out/thor_v1.

Order of the protocol: (1) hardware check P0 (Thor) vs the workstation pooled400 run on the same clips; (2) per precision
P in {P0, P1, P2, P3}: tau_eff(E_k, P) on P's own Heun-32 curve and the residuals of the paper metrics
(matched_temperature_stats, same per-clip table and interpolation); (3) paired shift dTau_eff(E_k) = tau_eff(P) - tau_eff(P0)
with the clips resampled jointly across precisions (2000 resamples, seed 20261029, tau_eff re-solved in every resample),
TOST on the 90% interval; (4) latency (T1) from out/thor_v1/latency/<variant>/summary.json; (5) verdicts T1-T5.

Usage: python -m frost.analysis.thor_analysis --thor THOR_COPY --out artifacts/thor_v1_analysis_<date>
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.core.matched_temperature_stats import METRICS, MARGINS, per_clip, summarize

TAUS=[1.,.9,.8,.7,.6,.5,.4];CURVE=[f'heun32_t{t}' for t in TAUS];REFERENCE='heun32_t1.0'
SAMPLERS=[f'euler{k}_t1.0' for k in [1,2,4,6,10]];CONFIGS=SAMPLERS+CURVE
TAU_MARGIN=.03  # new margin for dTau_eff (section 5)
BOOT_SEED=20261029
PROBES={'P0':'probe_bf16','P1':'P1','P2':'P2','P3':'P3'}


def tables(plans,names,seeds,data,records):
    human={n:np.load(data/f'{n}.npz')['ego_future_xyz'] for n in names}
    return per_clip(plans,names,CONFIGS,seeds,REFERENCE,human,{n:records[n]['speed_mps'] for n in names})


def arrays(table,names):
    """table[config][clip][metric] -> {config: {metric: array over names}} for vectorized bootstrap means."""
    return {c:{m:np.array([table[c][n][m] for n in names]) for m in ['spread',*METRICS]} for c in CONFIGS}


def matched(arr,idx,sampler,curve=CURVE):
    """Vectorized summarize(): same interpolation of the mean spread on the Heun-32 tau grid (idx: clip indices)."""
    mean=lambda c,m:float(arr[c][m][idx].mean())
    spreads=np.array([mean(c,'spread') for c in curve]);o=np.argsort(spreads);s=mean(sampler,'spread')
    out={'tau_eff':float(np.interp(s,spreads[o],np.array(TAUS)[o])),'in_range':bool(spreads.min()<=s<=spreads.max()),'spread':s}
    for m in METRICS:
        c=np.array([mean(x,m) for x in curve]);v=mean(sampler,m)
        out[m]={'value':v,'effect':v-mean(curve[0],m),'residual':v-float(np.interp(s,spreads[o],c[o]))}
    return out


def ci(xs,q=(.025,.975)):return np.quantile(np.asarray(xs),q).tolist()


def tost(boots,margin):
    lo,hi=np.quantile(np.asarray(boots),[.05,.95]);return {'ci90':[float(lo),float(hi)],'equivalent':bool(-margin<lo and hi<margin)}


def plan_diffs(a,b,names,seeds):
    """Per clip: per-plan max |delta| over points and xyz between two probes, for every seed x config both have."""
    per={}
    for n in names:
        d=[float(np.abs(a[k]-b[k]).max()) for s in seeds for c in CONFIGS for k in [f'{n}_s{s}_{c}'] if k in a and k in b]
        if d:per[n]=d
    return per


def load_probe(path):
    seeds=json.loads((path/'manifest.json').read_text())['seeds'] if (path/'manifest.json').exists() else list(range(9201,9207))
    with np.load(path/'plans.npz') as z:plans={k:z[k] for k in z.files}
    return plans,seeds,(path/'manifest.json').exists()


def reasoning_texts(path):
    out={}
    if (path/'reasoning.jsonl').exists():
        for line in (path/'reasoning.jsonl').read_text().splitlines():
            r=json.loads(line);out[r['clip']]=json.dumps(r['extra'],sort_keys=True)  # last record per clip wins (resume)
    return out


def latency(thor):
    out={}
    for v in PROBES:
        f=thor/'latency'/v/'summary.json'
        if f.exists():out[v]=json.loads(f.read_text())
    return out


def t1_verdict(lat):
    if 'P0' not in lat or 'call_ms' not in lat['P0'].get('batch1',{}):return {'verdict':'missing','reason':'no P0 latency summary'}
    b=lat['P0']['batch1'];call=b['call_ms']['median'];plan={s:v['plan_s']['median'] for s,v in b['plan'].items()}
    within={s:abs(plan[s]-b['plan'][s]['nfe']*call/1e3)/(b['plan'][s]['nfe']*call/1e3)<=.05 for s in plan}
    fits={s:plan[s]<.1 for s in plan}
    parts={'call_65_75ms':65<=call<=75,'plan_eq_nfe_x_call_5pct':all(within.values()),'only_E1_fits_100ms':fits.get('euler1')is True and not any(v for s,v in fits.items() if s!='euler1')}
    r={'p0_call_ms_median':call,'p0_plan_s_median':plan,'within_5pct':within,'fits_100ms':fits,**parts}
    if 'P1' in lat and 'call_ms' in lat['P1'].get('batch1',{}):
        red=1-lat['P1']['batch1']['call_ms']['median']/call;r['p1_call_reduction']=red;parts['p1_call_reduction_ge_25pct']=red>=.25
    else:parts['p1_call_reduction_ge_25pct']=None
    r['parts']=parts;r['verdict']='pass' if all(v is True for v in parts.values()) else ('incomplete' if any(v is None for v in parts.values()) else 'fail')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thor',type=Path,required=True,help='copy of the Thor output directory <FROST_THOR_ROOT>/out/thor_v1')
    p.add_argument('--workstation',type=Path,default=Path('artifacts/temperature_probe_a15_heldout_pooled400_v3_20261028'))
    p.add_argument('--data',type=Path,default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--range',type=int,nargs=2,default=[0,60]);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    records={r['name']:r for r in json.loads((a.data/'records.json').read_text()) if 'error' not in r}
    want=[f'context_{i:03d}' for i in range(*a.range)]
    probes={}
    for v,d in PROBES.items():
        if (a.thor/d/'plans.npz').exists():probes[v]=load_probe(a.thor/d)
    assert 'P0' in probes,'P0 (probe_bf16) is required'
    ws_plans,ws_seeds,_=load_probe(a.workstation)
    seeds=probes['P0'][1];assert all(pr[1]==seeds for pr in probes.values()) and ws_seeds==seeds
    has=lambda plans,n:all(f'{n}_s{s}_{c}' in plans for s in seeds for c in CONFIGS)
    names=[n for n in want if has(probes['P0'][0],n) and has(ws_plans,n)]
    res={'protocol':'protocols/thor.md (registered 29/10/2026 14:59 PDT)','range':a.range,'boot':a.boot,'boot_seed':BOOT_SEED,
        'margins':{**MARGINS,'tau_eff':TAU_MARGIN},'clips_requested':len(want),'probes':{v:{'complete':pr[2],'clips':sum(has(pr[0],n) for n in want)} for v,pr in probes.items()}}
    rng=np.random.default_rng(BOOT_SEED)

    # (1) Hardware check, first: P0 on Thor vs the workstation on the same clips.
    arr_ws=arrays(tables(ws_plans,names,seeds,a.data,records),names)
    arr={'P0':arrays(tables(probes['P0'][0],names,seeds,a.data,records),names)}  # recomputed below with the summarize check
    samples=[np.arange(len(names))]+[rng.integers(0,len(names),len(names)) for _ in range(a.boot)]
    per=plan_diffs(probes['P0'][0],ws_plans,names,seeds);clip_med=[float(np.median(per[n])) for n in names]
    hw={'clips':len(names),'max_abs_dplan_m':{'median_of_clip_medians':float(np.median(clip_med)),'median_all_plans':float(np.median([x for n in names for x in per[n]])),
        'p90_all_plans':float(np.quantile([x for n in names for x in per[n]],.9)),'clips_with_any_plan_gt_0.10m':int(sum(max(per[n])>.1 for n in names)),
        'per_clip_median':dict(zip(names,clip_med))},'tau_eff':{}}
    for s in ['euler10_t1.0','euler4_t1.0']:
        th=[matched(arr['P0'],i,s)['tau_eff'] for i in samples];ws=[matched(arr_ws,i,s)['tau_eff'] for i in samples];d=np.subtract(th,ws)
        hw['tau_eff'][s]={'thor':th[0],'workstation':ws[0],'diff':float(d[0]),'diff_ci95':ci(d[1:])}
    res['hardware']=hw

    # (2) Matched-temperature test per precision against its own curve, all on the same paired clip resamples.
    paired={v:[n for n in names if has(plans,n)] for v,(plans,_,_) in probes.items()}
    full=[v for v in probes if paired[v]==names]  # precisions covering every clip: resampled jointly with P0
    res['precisions']={};runs={}
    for v in full:
        table=tables(probes[v][0],names,seeds,a.data,records);arr[v]=arrays(table,names)
        runs[v]={s:[matched(arr[v],i,s) for i in samples] for s in SAMPLERS+[REFERENCE]}
        for s in SAMPLERS:  # reuse check: the vectorized point estimate equals matched_temperature_stats.summarize
            ref=summarize(table,names,CURVE,TAUS,s);pt=runs[v][s][0]
            assert abs(ref['tau_eff']-pt['tau_eff'])<1e-9 and all(abs(ref[m]['residual']-pt[m]['residual'])<1e-9 for m in METRICS),(v,s)
        entry={}
        for s in SAMPLERS:
            r=runs[v][s];pt=r[0];bt=r[1:];e={'tau_eff':[pt['tau_eff'],*ci([b['tau_eff'] for b in bt])],'in_range':pt['in_range']}
            for m in METRICS:
                rs=[b[m]['residual'] for b in bt];e[m]={'value':pt[m]['value'],'effect':[pt[m]['effect'],*ci([b[m]['effect'] for b in bt])],
                    'residual':[pt[m]['residual'],*ci(rs)],**tost(rs,MARGINS[m])}
            entry[s]=e
        res['precisions'][v]=entry
    res['precisions_incomplete']={v:len(paired[v]) for v in probes if v not in full}

    # (3) Paired shifts vs P0: dTau_eff(E_k) and the native temperature of Heun-32 tau=1 (its spread read on P0's curve).
    res['shift_vs_P0']={}
    for v in [x for x in full if x!='P0']:
        sh={}
        for s in SAMPLERS:
            d=np.array([x['tau_eff']-y['tau_eff'] for x,y in zip(runs[v][s],runs['P0'][s])])
            sh[s]={'dtau_eff':float(d[0]),'ci95':ci(d[1:]),**tost(d[1:],TAU_MARGIN)}
        nat=[matched({**arr['P0'],'native':arr[v][REFERENCE]},i,'native') for i in samples]
        sh['heun32_native_tau']={'tau_on_P0_curve':nat[0]['tau_eff'],'ci95':ci([x['tau_eff'] for x in nat[1:]]),'in_range':nat[0]['in_range'],
            'dtau':nat[0]['tau_eff']-1.,'spread_ratio_vs_P0':nat[0]['spread']/float(arr['P0'][REFERENCE]['spread'].mean())}
        res['shift_vs_P0'][v]=sh

    # Reasoning text of P2 vs the bf16 VLM (P1 runs the same bf16 VLM with offload as P0 and logs its text).
    tx={v:reasoning_texts(a.thor/PROBES[v]) for v in ['P1','P2','P3']}
    common=[n for n in names if n in tx['P1'] and n in tx['P2']]
    res['reasoning_P2_vs_bf16']={'clips':len(common),'fraction_different':float(np.mean([tx['P1'][n]!=tx['P2'][n] for n in common])) if common else None}
    lin={}
    for v in ['P1','P2','P3']:
        f=a.thor/v/'linear_errors.json'
        if f.exists():lin[v]=json.loads(f.read_text())['summary']
    res['linear_errors']=lin;res['latency']=latency(a.thor)

    # (4) Verdicts.
    V={}
    V['T1']=t1_verdict(res['latency'])
    if 'P1' in res['shift_vs_P0']:
        t2={s:res['shift_vs_P0']['P1'][s] for s in ['euler10_t1.0','euler4_t1.0']}
        V['T2']={'parts':{s:t['equivalent'] for s,t in t2.items()},'verdict':'pass' if all(t['equivalent'] for t in t2.values()) else 'fail',
            'falsified':[s for s,t in t2.items() if not t['equivalent'] and (t['ci95'][0]>0 or t['ci95'][1]<0) and abs(t['dtau_eff'])>TAU_MARGIN],
            'inconclusive':[s for s,t in t2.items() if not t['equivalent'] and t['ci95'][0]<=0<=t['ci95'][1]]}
        p1=res['precisions']['P1'];parts={f'{s}_ade1_residual_equiv':p1[s]['ade1']['equivalent'] for s in ['euler10_t1.0','euler4_t1.0']}
        parts['E10_tau_eff_in_0.80_0.90']=.80<=p1['euler10_t1.0']['tau_eff'][0]<=.90
        V['T3']={'parts':parts,'verdict':'pass' if all(parts.values()) else 'fail',
            'P0_reference':{s:res['precisions']['P0'][s]['ade1']['equivalent'] for s in ['euler10_t1.0','euler4_t1.0']}}
    else:V['T2']=V['T3']={'verdict':'missing','reason':'P1 incomplete or absent'}
    parts={'median_max_dplan_lt_0.10m':hw['max_abs_dplan_m']['median_of_clip_medians']<.10,'E10_tau_eff_diff_lt_0.02':abs(hw['tau_eff']['euler10_t1.0']['diff'])<.02}
    V['T4']={'parts':parts,'verdict':'pass' if all(parts.values()) else 'fail'}
    if 'P2' in res['shift_vs_P0']:
        t=res['shift_vs_P0']['P2']['euler10_t1.0']
        V['T5']={'equivalent':t['equivalent'],'verdict':'pass' if t['equivalent'] else 'fail','reasoning_fraction_different':res['reasoning_P2_vs_bf16']['fraction_different']}
    else:V['T5']={'verdict':'missing','reason':'P2 incomplete or absent'}
    if 'P3' in res['shift_vs_P0']:V['P3_exploratory']={s:res['shift_vs_P0']['P3'][s] for s in ['euler10_t1.0','euler4_t1.0']}
    res['verdicts']=V

    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=2))
    (a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(f"== hardware ({len(names)} clips): median max|dplan| {hw['max_abs_dplan_m']['median_of_clip_medians']:.3f} m; "
        +'; '.join(f"tau_eff {s.split('_')[0]} thor {x['thor']:.3f} ws {x['workstation']:.3f} diff {x['diff']:+.3f}" for s,x in hw['tau_eff'].items()))
    for v,e in res['precisions'].items():
        print(f'== {v}')
        for s,x in e.items():print(f"  {s:13s} tau_eff {x['tau_eff'][0]:.3f} [{x['tau_eff'][1]:.3f},{x['tau_eff'][2]:.3f}] | ade1 res {x['ade1']['residual'][0]:+.3f}{'≡' if x['ade1']['equivalent'] else ''} | flip res {x['flip']['residual'][0]:+.3f}{'≡' if x['flip']['equivalent'] else ''}")
    for v,sh in res['shift_vs_P0'].items():
        print(f'== {v} - P0: '+'; '.join(f"{s.split('_')[0]} {x['dtau_eff']:+.3f} [{x['ci90'][0]:+.3f},{x['ci90'][1]:+.3f}]{'≡' if x['equivalent'] else ''}" for s,x in sh.items() if s in SAMPLERS))
    for k,v in V.items():print(k,v.get('verdict',v))


if __name__=='__main__':main()
