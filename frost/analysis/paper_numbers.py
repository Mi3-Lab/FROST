"""Compute every number the paper quotes from the temperature probes, on one sample and one reference.

Sample: the same 60 PhysicalAI-AV test clips (context_100-159) and six seeds for every model;
reference: Heun-32 started from tau*z. A shrinkage control (shrink{k}) moves each reference sample
linearly toward the one-step mean, E1 + c (A1 - E1), with c solved so that its mean spread equals E_k's on
the full sample (E1 itself keeps some seed spread in trained networks, so c is not the plain spread ratio); it is a sampler with no integrator that the matched test should not distinguish from E_k.
Clips are the unit: all intervals come from a bootstrap
that resamples clips jointly across models (the models share the clips). Writes an artifact
summary and paper/tables/numbers.tex with one macro per quoted number.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions
from frost.core.temperature_report import entropy

MODELS=[('alpamayo15','AOneFive','temperature_probe_v1_20261022'),('alpamayo_r1','ROne','temperature_probe_r1_v1_20261022'),
    ('alpamayo2','ATwo','temperature_probe_a2_v1_20261022')]
TAUS=[1.,.9,.8,.7,.6,.5,.4];CURVE=[f'heun32_t{t}' for t in TAUS];REFERENCE='heun32_t1.0'
MARGINS={'ade1':.05,'flip':.02}


def clip_metrics(plans,name,config,seeds,speed,human,ref_dec):
    trajs=[plans[f'{name}_s{s}_{config}'] for s in seeds];d=[decisions(t,speed) for t in trajs];key=lambda x:(x['longitudinal'],x['lateral'])
    mean=np.mean([plans[f'{name}_s{s}_euler1_t1.0'] for s in seeds],0)[:,:2]
    contraction=[np.linalg.norm(t[:,:2]-plans[f'{name}_s{s}_euler1_t1.0'][:,:2])/max(np.linalg.norm(plans[f'{name}_s{s}_{REFERENCE}'][:,:2]-plans[f'{name}_s{s}_euler1_t1.0'][:,:2]),1e-9) for s,t in zip(seeds,trajs)]
    return {'spread':float(np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]])),
        'ade1':float(np.mean([np.linalg.norm(t[:,:2]-human[:,:2],axis=1).mean() for t in trajs])),
        'minade6':float(min(np.linalg.norm(t[:,:2]-human[:,:2],axis=1).mean() for t in trajs)),
        'flip':float(np.mean([key(a)!=key(b) for a,b in zip(d,ref_dec)])),'entropy':entropy([key(x) for x in d]),
        'stops':int(sum(x['longitudinal']=='stop' for x in d)),'contraction':contraction,'del_mean':float(np.linalg.norm(np.mean(trajs,0)[:,:2]-mean,axis=1).mean())}


def spread(trajs):return float(np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]]))


def add_shrinkage(plans,names,seeds,steps=(4,10)):
    shrunk=lambda n,s,c:plans[f'{n}_s{s}_euler1_t1.0']+c*(plans[f'{n}_s{s}_{REFERENCE}']-plans[f'{n}_s{s}_euler1_t1.0'])
    for k in steps:
        target=np.mean([spread([plans[f'{n}_s{s}_euler{k}_t1.0'] for s in seeds]) for n in names]);lo,hi=0.,1.
        for _ in range(30):
            c=(lo+hi)/2
            if np.mean([spread([shrunk(n,s,c) for s in seeds]) for n in names])<target:lo=c
            else:hi=c
        for n in names:
            for s in seeds:
                plans[f'{n}_s{s}_shrink{k}_t1.0']=shrunk(n,s,c)


def load(args):
    records={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    names=[f'context_{i:03d}' for i in range(*args.range)];tables={}
    for key,_,probe in MODELS:
        path=args.artifacts/probe;seeds=json.loads((path/'manifest.json').read_text())['seeds']
        with np.load(path/'plans.npz') as z:plans={k:z[k] for k in z.files}
        add_shrinkage(plans,names,seeds)
        configs=sorted({k.split('_',3)[-1] for k in plans if '_s92' in k});table=collections.defaultdict(dict)
        for n in names:
            human=np.load(args.data/f'{n}.npz')['ego_future_xyz'];speed=records[n]['speed_mps']
            ref_dec=[decisions(plans[f'{n}_s{s}_{REFERENCE}'],speed) for s in seeds]
            for c in configs:table[c][n]=clip_metrics(plans,n,c,seeds,speed,human,ref_dec)
        tables[key]=table
    return names,tables


def matched(table,sample,sampler):
    mean=lambda c,m:float(np.mean([table[c][n][m] for n in sample]))
    spreads=np.array([mean(c,'spread') for c in CURVE]);o=np.argsort(spreads);s=mean(sampler,'spread')
    out={'tau_eff':float(np.interp(s,spreads[o],np.array(TAUS)[o])),'in_range':bool(spreads.min()<=s<=spreads.max())}
    for m in ['ade1','minade6','flip','entropy']:
        curve=np.array([mean(c,m) for c in CURVE]);at=float(np.interp(s,spreads[o],curve[o]));v=mean(sampler,m);base=mean(REFERENCE,m)
        out[m]={'reference':base,'sampler':v,'matched':at,'effect':v-base,'residual':v-at}
    return out


def signature(table,sample,sampler,shrink):
    """Per-sample contraction toward E1 at matched spread: medians and contracted fractions of the sampler, the
    reference solver at the sampler's spread (interpolated on the tau grid) and the spread-matched shrinkage."""
    pool=lambda c:np.array([r for n in sample for r in table[c][n]['contraction']])
    spreads=np.array([np.mean([table[c][n]['spread'] for n in sample]) for c in CURVE]);o=np.argsort(spreads)
    curve=[pool(c) for c in CURVE];s=np.mean([table[sampler][n]['spread'] for n in sample]);e=pool(sampler);sh=pool(shrink)
    temp_med=float(np.interp(s,spreads[o],np.array([np.median(r) for r in curve])[o]))
    temp_frac=float(np.interp(s,spreads[o],np.array([np.mean(r<1) for r in curve])[o]))
    return {'euler_median':float(np.median(e)),'temp_median':temp_med,'shrink_median':float(np.median(sh)),
        'euler_frac':float(np.mean(e<1)),'temp_frac':temp_frac}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--macros',type=Path,required=True);p.add_argument('--range',type=int,nargs=2,default=[100,160])
    p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--probes',nargs='*',default=None,help='key=probe_dir: analyze only these models from these probe directories (e.g. held-out runs)')
    args=p.parse_args()
    if args.probes:
        global MODELS
        tags={k:t for k,t,_ in MODELS};MODELS=[(k,tags[k],d) for k,d in (x.split('=',1) for x in args.probes)]
    names,tables=load(args);rng=np.random.default_rng(20261026)
    samples=[names]+[list(rng.choice(names,len(names))) for _ in range(args.boot)]
    result={'clips':len(names),'boot':args.boot,'reference':REFERENCE,'margins':MARGINS,'models':{},'stops':{}}
    for key,_,_ in MODELS:
        table=tables[key];samplers=[c for c in table if (c.startswith(('euler','shrink')) or c in ('heun3_t1.0','heun5_t1.0')) and c.endswith('_t1.0')]
        model={}
        for sampler in samplers:
            runs=[matched(table,s,sampler) for s in samples];point=runs[0];boots=runs[1:]
            entry={'tau_eff':[point['tau_eff'],*np.quantile([b['tau_eff'] for b in boots],[.025,.975]).tolist()],'in_range':point['in_range']}
            for m in ['ade1','minade6','flip','entropy']:
                eff=np.array([b[m]['effect'] for b in boots]);res=np.array([b[m]['residual'] for b in boots])
                entry[m]={**{k:point[m][k] for k in ['reference','sampler','matched']},
                    'effect':[point[m]['effect'],*np.quantile(eff,[.025,.975]).tolist()],
                    'residual':[point[m]['residual'],*np.quantile(res,[.025,.975]).tolist()],
                    'fraction_explained':float(1-point[m]['residual']/point[m]['effect']) if abs(point[m]['effect'])>1e-9 else None}
                if m in MARGINS:
                    lo,hi=np.quantile(res,[.05,.95]);entry[m]['equivalent']=bool(-MARGINS[m]<lo and hi<MARGINS[m])
            ratios=[r for n in names for r in table[sampler][n]['contraction']]
            entry['contraction']={'median':float(np.median(ratios)),'fraction_below_one':float(np.mean(np.array(ratios)<1)),'n':len(ratios)}
            # Per-sample signature: the reference solver's contraction at the sampler's spread (interpolated on the tau grid).
            spreads=np.array([np.mean([table[c][n]['spread'] for n in names]) for c in CURVE]);o=np.argsort(spreads)
            curve=[np.array([r for n in names for r in table[c][n]['contraction']]) for c in CURVE];s_=np.mean([table[sampler][n]['spread'] for n in names])
            entry['contraction']['matched_median']=float(np.interp(s_,spreads[o],np.array([np.median(r) for r in curve])[o]))
            entry['contraction']['matched_fraction_below_one']=float(np.interp(s_,spreads[o],np.array([np.mean(r<1) for r in curve])[o]))
            model[sampler]=entry
        for k in (4,10):
            runs=[signature(table,s,f'euler{k}_t1.0',f'shrink{k}_t1.0') for s in samples]
            model[f'euler{k}_t1.0']['signature']={'point':runs[0],'boot':runs[1:]}
        result['models'][key]=model
        result['stops'][key]={c:{'plans':int(sum(table[c][n]['stops'] for n in names)),
            'clips':sorted(n for n in names if table[c][n]['stops']>0)} for c in ['heun32_t1.0','heun32_t0.6','heun32_t0.4','euler10_t1.0','euler4_t1.0','euler1_t1.0']}
    diffs={'euler_minus_temp_median':lambda r:r['euler_median']-r['temp_median'],'euler_minus_shrink_median':lambda r:r['euler_median']-r['shrink_median'],
        'euler_minus_temp_frac':lambda r:r['euler_frac']-r['temp_frac']}
    result['signature']={}
    for k in (4,10):
        sig=[result['models'][key][f'euler{k}_t1.0']['signature'] for key,_,_ in MODELS];out={}
        for name,f in diffs.items():
            boots=np.mean([[f(b) for b in s['boot']] for s in sig],0)
            out[name]=[float(np.mean([f(s['point']) for s in sig])),*np.quantile(boots,[.025,.975]).tolist()]
            for key,s in zip([m for m,_,_ in MODELS],sig):out.setdefault('per_model',{}).setdefault(key,{})[name]=[f(s['point']),*np.quantile([f(b) for b in s['boot']],[.025,.975]).tolist()]
        result['signature'][f'E{k}']=out
        for s in sig:s.pop('boot')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    macros=[]
    def put(name,value):macros.append(f'\\newcommand{{\\{name}}}{{{value}}}')
    put('NClips',len(names))
    e1=[np.mean([tables[key]['euler1_t1.0'][n]['spread'] for n in names])/np.mean([tables[key][REFERENCE][n]['spread'] for n in names]) for key,_,_ in MODELS]
    put('EOneSpreadMin',f'{100*min(e1):.0f}');put('EOneSpreadMax',f'{100*max(e1):.0f}')
    fmt=lambda x,d=2:f'{x:.{d}f}'.replace('-','\\text{-}') if x<0 else f'{x:.{d}f}'
    for key,tag,_ in MODELS:
        m=result['models'][key]
        for k in [4,10]:
            e=m[f'euler{k}_t1.0'];base=f'{tag}E{"Four" if k==4 else "Ten"}'
            put(base+'Tau',fmt(e['tau_eff'][0]));put(base+'TauLo',fmt(e['tau_eff'][1]));put(base+'TauHi',fmt(e['tau_eff'][2]))
            put(base+'Contraction',fmt(e['contraction']['median']));put(base+'FracContracted',f"{100*e['contraction']['fraction_below_one']:.0f}")
            put(base+'ExplAde',f"{100*e['ade1']['fraction_explained']:.0f}");put(base+'ExplFlip',f"{100*e['flip']['fraction_explained']:.0f}")
        for h in ['heun3_t1.0','heun5_t1.0']:
            if h in m:put(f'{tag}{"HeunThree" if h.startswith("heun3") else "HeunFive"}FlipResid',f"{100*m[h]['flip']['residual'][0]:.1f}")
        s=result['stops'][key];put(f'{tag}StopsRef',s['heun32_t1.0']['plans']);put(f'{tag}StopsEFour',s['euler4_t1.0']['plans']);put(f'{tag}StopsLow',s['heun32_t0.4']['plans'])
    stop_clips=sorted({c for key in result['stops'] for v in result['stops'][key].values() for c in v['clips']})
    put('StopClips',len(stop_clips));put('StopPlans',6*len(names));result['stop_clips']=stop_clips
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    args.macros.write_text('% Generated by frost/analysis/paper_numbers.py; do not edit by hand.\n'+'\n'.join(macros)+'\n')
    for key in result['models']:
        print('==',key)
        for s,e in result['models'][key].items():
            print(f"  {s:13s} tau {e['tau_eff'][0]:.2f}{'' if e['in_range'] else ' (out of range)'} contr {e['contraction']['median']:.2f} ({100*e['contraction']['fraction_below_one']:.0f}% <1) | ADE expl {e['ade1']['fraction_explained']} eq {e['ade1'].get('equivalent')} | flip resid {100*e['flip']['residual'][0]:+.1f} [{100*e['flip']['residual'][1]:+.1f},{100*e['flip']['residual'][2]:+.1f}] eq {e['flip'].get('equivalent')}")
    print('stop clips',stop_clips,json.dumps(result['stops'],indent=0)[:600])


if __name__=='__main__':main()
