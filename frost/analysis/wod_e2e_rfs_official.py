"""Registered WOD-E2E analysis re-run with the OFFICIAL Waymo Rater Feedback Score in place of our RFS-proxy.

The official module (waymo-open-dataset, src/waymo_open_dataset/metrics/python/rater_feedback_utils.py) is not
redistributed: download that single file unmodified (THIRD_PARTY.md) and point $WAYMO_RFS_OFFICIAL at it. It is loaded by
explicit file path with importlib after its sha256 is checked. It imports numpy only.

Input mapping to ``get_rater_feedback_score`` (one call per segment x config, B = 1):
* inference_trajectories [1, I=6, 20, 2]: the 6 seed plans (9201-9206) of the config. Alpamayo plans are 64 waypoints at
  10 Hz (t = 0.1..6.4 s, vehicle frame at t = 0); they are resampled to the official 4 Hz grid t = 0.25 k, k = 1..20, by
  linear interpolation in x and y. Times are handled in integer units of 0.05 s so that the two points the metric uses
  (index 11 = 3.0 s, index 19 = 5.0 s) are exactly plan waypoints 29 and 49, not interpolated. Only x, y are passed.
* inference_probs [1, 6] = 1/6 each. The official score is sum_i p_i score_i, i.e. exactly the 6-seed mean that the
  registered analysis uses per segment.
* rater_specified_trajectories: list (B) of list (P = 3) of arrays [T_p, 2]: ``rater_xy`` from the context npz (21
  waypoints at 4 Hz from t = 0.25 s), with the NaN rows of our converter's padding (protocol addendum 1) removed, so short
  rater paths have their true length T_p in 2..21. NaN padding is checked to be trailing only. The official code then
  truncates to 20 waypoints and pads short paths by repeating the last waypoint.
* rater_feedback_labels: list of [P] arrays = ``rater_scores`` (no label is -1 in our 479 contexts; checked).
* init_speed [1] = records.json ``speed_mps`` = |past_states vel[-1]| (the npz ``wod_speed``), the same quantity the
  official colab passes.
* All other arguments at their defaults: multipliers (1, 4), decay 0.1, 4 Hz, 5 s, 3 rater trajectories, floor 4.
All arrays are cast to float64 before the call.

Everything else is the registered pipeline (frost.analysis.wod_e2e_analysis): same plans, same spread / ADE_1 / flips, tau_eff by
spread matching on the Heun-32 curve, residual vs the curve at tau_eff, segment bootstrap (seed 20261029) re-solving
tau_eff, TOST at 90% with margins RFS 0.08, ADE_1 0.05 m, flips 2 pt. Our proxy score is computed alongside for agreement.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from frost.analysis import wod_e2e_analysis as A
from frost.analysis.wod_e2e_rfs import rfs as rfs_ours

OFFICIAL=Path(os.environ.get('WAYMO_RFS_OFFICIAL','third_party/waymo_rfs_official/rater_feedback_utils.py'))
OFFICIAL_SHA256='fe994a096141957b8f936d613b08876bee60ec5b746749b1a79331217f4edee9'
GRID=5*np.arange(1,21)  # official 4 Hz grid, units of 0.05 s


def load_official(path=OFFICIAL):
    b=Path(path).read_bytes();h=hashlib.sha256(b).hexdigest()
    if h!=OFFICIAL_SHA256:raise RuntimeError(f'official RFS file sha256 {h} != pinned {OFFICIAL_SHA256}')
    spec=importlib.util.spec_from_file_location('waymo_rater_feedback_utils_official',str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def to_4hz(xy,times_s):
    """Resample a (T,2) path given at times_s (seconds) to the official grid t = 0.25..5.0 s."""
    u=np.rint(np.asarray(times_s,float)/.05);assert np.allclose(u*.05,times_s)
    return np.stack([np.interp(GRID,u,xy[:,0]),np.interp(GRID,u,xy[:,1])],1)


def raters_list(rater_xy):
    out=[]
    for r in np.asarray(rater_xy,np.float64)[:,:,:2]:
        k=~np.isnan(r).any(1);n=int(k.sum());assert n>0 and k[:n].all() and not k[n:].any(),'NaN padding must be trailing'
        out.append(r[:n])
    return out


def official_score(M,preds_4hz,rater_xy,rater_scores,speed):
    """preds_4hz (I,20,2). Returns (weighted score, per-inference scores, per-inference fully-within-trust-region flag)."""
    P=np.asarray(preds_4hz,np.float64)[None];I=P.shape[1]
    o=M.get_rater_feedback_score(P,np.full((1,I),1./I),[raters_list(rater_xy)],[np.asarray(rater_scores,np.float64)],np.array([float(speed)],np.float64))
    # per-inference scores: recompute with one-hot probs is wasteful; the module returns only the weighted sum, so call per seed
    per=np.array([M.get_rater_feedback_score(P[:,[i]],np.ones((1,1)),[raters_list(rater_xy)],[np.asarray(rater_scores,np.float64)],np.array([float(speed)]))['rater_feedback_score'][0] for i in range(I)])
    assert abs(per.mean()-o['rater_feedback_score'][0])<1e-9
    return float(o['rater_feedback_score'][0]),per,o['is_fully_within_trust_region'][0]


def spearman(x,y):
    r=lambda v:np.argsort(np.argsort(v,kind='stable'),kind='stable').astype(float)
    def avg(v):  # average ranks for ties
        rk=r(v);u,inv=np.unique(v,return_inverse=True)
        return np.array([rk[inv==j].mean() for j in range(len(u))])[inv]
    return float(np.corrcoef(avg(x),avg(y))[0,1])


def stats(T,names_ok,draws,a,CFG):
    N=len(names_ok);ALL=np.arange(N);MARG=A.MARGINS
    res={'configs':{c:{m:float(T[c][m].mean()) for m in ('rfs','ade1','ade_rater','spread','flip')} for c in CFG}}
    res['rfs_range_over_configs']=float(max(v['rfs'] for v in res['configs'].values())-min(v['rfs'] for v in res['configs'].values()))
    res['metric_uninformative']=bool(res['rfs_range_over_configs']<.1);res['samplers']={}
    for s in A.SAMPLERS:
        m=A.matched(T,ALL,s);B=[A.matched(T,d,s) for d in draws]
        m['tau_eff_ci95']=np.quantile([b['tau_eff'] for b in B],[.025,.975]).tolist();m['boot_in_range']=float(np.mean([b['in_range'] for b in B]))
        for q in MARG:
            r=[b[q]['residual'] for b in B];e=[b[q]['raw_effect'] for b in B]
            m[q]['residual_ci90']=np.quantile(r,[.05,.95]).tolist();m[q]['residual_ci95']=np.quantile(r,[.025,.975]).tolist()
            m[q]['equivalent']=bool(m['in_range'] and -MARG[q]<m[q]['residual_ci90'][0] and m[q]['residual_ci90'][1]<MARG[q])
            m[q]['raw_effect_ci95']=np.quantile(e,[.025,.975]).tolist();m[q]['raw_effect_includes_0']=bool(m[q]['raw_effect_ci95'][0]<=0<=m[q]['raw_effect_ci95'][1])
        m['_boot_rfs_residuals']=np.array([b['rfs']['residual'] for b in B])
        res['samplers'][s]=m
    d4=[T['heun32_t0.4']['rfs'][d].mean()-T[A.REF]['rfs'][d].mean() for d in draws];pt=float(T['heun32_t0.4']['rfs'].mean()-T[A.REF]['rfs'].mean())
    res['W5_detail']={'rfs_t0.4_minus_t1':pt,'ci95':np.quantile(d4,[.025,.975]).tolist(),'includes_0':bool(np.quantile(d4,.025)<=0<=np.quantile(d4,.975))}
    S=res['samplers'];e10,e4,e1=S['euler10_t1.0'],S['euler4_t1.0'],S['euler1_t1.0']
    res['W']={'W1':bool(e10['rfs']['equivalent'] and e4['rfs']['equivalent']),'W1_parts':{'E10':e10['rfs']['equivalent'],'E4':e4['rfs']['equivalent']},
        'W2':bool(e10['in_range'] and .75<=e10['tau_eff']<=.92 and e4['in_range'] and .5<=e4['tau_eff']<=.75 and e1['tau_eff']<=.45),
        'W3':bool(e10['ade1']['equivalent'] and e4['ade1']['equivalent']),'W4_E10':bool(e10['flip']['equivalent']),'W4_E4_E1_informational':[bool(e4['flip']['equivalent']),bool(e1['flip']['equivalent'])],
        'W5':bool(pt>=.05)}
    return res


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('artifacts/wod_e2e_val_contexts_v2_20261029'))
    p.add_argument('--probe',default='artifacts/wod_e2e_probe_a15_v1_20261029_p{}');p.add_argument('--shards',type=int,default=8)
    p.add_argument('--out',type=Path,default=Path('artifacts/wod_e2e_analysis_official_rfs_v1_20261030'))
    p.add_argument('--boot',type=int,default=10000);p.add_argument('--seed',type=int,default=A.SEED);p.add_argument('--samples',default='280,479')
    a=p.parse_args();M=load_official();CFG=A.CURVE+A.SAMPLERS
    records={r['name']:r for r in json.loads((a.data/'records.json').read_text())}
    plans={};seeds=None
    for i in range(a.shards):
        d=Path(a.probe.format(i));m=json.loads((d/'manifest.json').read_text());seeds=seeds or m['seeds'];assert m['seeds']==seeds
        with np.load(d/'plans.npz') as z:plans.update({k:z[k] for k in z.files})
    nmax=max(int(x) for x in a.samples.split(','));names=[f'context_{i:03d}' for i in range(nmax)];assert all('error' not in records[n] for n in names)
    G4=A.PT  # 10 Hz plan times, 0.1..6.4 s
    per={};refl={}
    for n in names:
        c=np.load(a.data/f'{n}.npz');ctx={k:c[k] for k in ('rater_xy','rater_scores','wod_future_xyz','ego_future_xyz')};v=records[n]['speed_mps']
        assert abs(float(c['wod_speed'])-v)<1e-5 and (ctx['rater_scores']>=0).all()
        per[n]=A.segment_table(plans,n,seeds,ctx,v)  # registered spread / ADE_1 / flips and our proxy 'rfs'
        for cf in CFG:
            tr=[plans[f'{n}_s{s}_{cf}'] for s in seeds];sc,ps,ins=official_score(M,np.stack([to_4hz(t[:,:2],G4) for t in tr]),ctx['rater_xy'],ctx['rater_scores'],v)
            per[n][cf]['rfs_ours']=per[n][cf].pop('rfs');per[n][cf]['rfs_official']=sc;per[n][cf]['rfs_official_per_seed']=ps.tolist();per[n][cf]['fully_within_trust_region']=ins.astype(bool).tolist()
        rl=raters_list(ctx['rater_xy']);pad=lambda r:np.concatenate([r,np.repeat(r[-1:],max(20-len(r),0),0)])[:20]
        refl[n]={'log_future':official_score(M,ctx['wod_future_xyz'][None,:,:2],ctx['rater_xy'],ctx['rater_scores'],v)[0],
            'best_rater':max(official_score(M,pad(r)[None],ctx['rater_xy'],ctx['rater_scores'],v)[0] for r in rl),'max_rater_score':float(ctx['rater_scores'].max()),
            'log_future_ours':rfs_ours(ctx['wod_future_xyz'][:,:2],A.FT,ctx['rater_xy'],ctx['rater_scores'])}
    out={'official_module':{'path':str(OFFICIAL.relative_to(OFFICIAL.parents[2])),'sha256':OFFICIAL_SHA256,'commit_last_touching_file':'e378e5ba5bb8460184e5c878c82c9e593bdb2927','master_head_at_download':'99a4cb3ff07e2fe06c2ce73da001f850f628e45a'},
        'input_mapping':__doc__.split('Input mapping')[1].split('Everything else')[0].strip(),'seeds':seeds,'boot':a.boot,'boot_seed':a.seed,'margins':A.MARGINS,'samples':{}}
    for N in [int(x) for x in a.samples.split(',')]:
        nm=names[:N];segs=[records[n]['segment'] for n in nm]
        rng=np.random.default_rng(a.seed);draws=[rng.integers(0,N,N) for _ in range(a.boot)]
        Tb={c:{m:np.array([per[n][c][m] for n in nm]) for m in ('ade1','ade_rater','spread','flip','rfs_official','rfs_ours')} for c in CFG}
        R={}
        for which in ('official','ours'):
            T={c:dict(Tb[c],rfs=Tb[c][f'rfs_{which}']) for c in CFG};R[which]=stats(T,nm,draws,a,CFG)
        r=R['official'];r['n_segments']=N;r['n_distinct_segments']=len(set(segs))
        r['reference_levels']={k:float(np.mean([refl[n][k] for n in nm])) for k in ('log_future','best_rater','max_rater_score','log_future_ours')}
        x=np.concatenate([Tb[c]['rfs_ours'] for c in CFG]);y=np.concatenate([Tb[c]['rfs_official'] for c in CFG])
        ag={'pooled_all_configs':{'n':len(x),'pearson':float(np.corrcoef(x,y)[0,1]),'spearman':spearman(x,y),'mean_shift_official_minus_ours':float((y-x).mean())}}
        for c in CFG:
            xo,yo=Tb[c]['rfs_ours'],Tb[c]['rfs_official'];ag[c]={'pearson':float(np.corrcoef(xo,yo)[0,1]),'spearman':spearman(xo,yo),'mean_shift':float((yo-xo).mean()),'n_abs_diff_gt_0.5':int((abs(yo-xo)>.5).sum())}
        ag['per_seed_scores_below_floor_official']=int(sum((np.array(per[n][c]['rfs_official_per_seed'])<4-1e-12).sum() for n in nm for c in CFG))
        r['agreement_official_vs_ours']=ag
        cmp={}
        for s in A.SAMPLERS:
            o,u=R['official']['samplers'][s],R['ours']['samplers'][s];dd=o.pop('_boot_rfs_residuals')-u.pop('_boot_rfs_residuals')
            cmp[s]={'tau_eff':o['tau_eff'],'rfs_residual_official':o['rfs']['residual'],'rfs_residual_ours':u['rfs']['residual'],'ci90_official':o['rfs']['residual_ci90'],'ci90_ours':u['rfs']['residual_ci90'],
                'equivalent_official':o['rfs']['equivalent'],'equivalent_ours':u['rfs']['equivalent'],'raw_effect_official':o['rfs']['raw_effect'],'raw_effect_ours':u['rfs']['raw_effect'],
                'residual_diff_official_minus_ours':o['rfs']['residual']-u['rfs']['residual'],'residual_diff_ci95':np.quantile(dd,[.025,.975]).tolist()}
        r['ours_recomputed']={'W':R['ours']['W'],'W5_detail':R['ours']['W5_detail']}
        r['comparison_rfs']=cmp;r['verdict_changes']={k:{'official':r['W'][k],'ours':R['ours']['W'][k]} for k in r['W'] if r['W'][k]!=R['ours']['W'][k]}
        out['samples'][str(N)]=r
        print(N,json.dumps({'W':r['W'],'changes':r['verdict_changes'],'W5':r['W5_detail'],'cmp':cmp,'agree':ag['pooled_all_configs'],'levels':r['reference_levels']},indent=1))
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'summary.json').write_text(json.dumps(out,indent=1))
    (a.out/'per_segment.json').write_text(json.dumps({'per_segment':per,'reference':refl},indent=1));(a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
