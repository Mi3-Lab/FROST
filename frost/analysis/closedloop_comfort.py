"""Hard braking and lane departure in the closed-loop grid (protocols/closed_loop_alpasim.md, addendum 3).

Per rollout: ego xy from rollout.asl actor poses -> speed by finite differences, 0.5 s moving average,
longitudinal acceleration (same smoothing) and jerk; a hard-braking event is acceleration <= -3.0 m/s^2
held for >= 0.3 s. Lane departure: share of eval-relevant steps with min_distance_to_lane_boundary_m == 0,
and whether such a run lasts >= 1 s. Scene-paired McNemar for binary metrics, scene bootstrap for
continuous ones, on the C1-C3 pairs. Run with the AlpaSim venv (polars, alpasim_utils).
"""
import argparse
import asyncio
import glob
import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import binomtest
from alpasim_utils.logs import async_read_pb_log

PAIRS=[('heun32_t0.65','heun32_t1.0','C1'),('euler4_t1.0','heun32_t0.65','C2 matched'),('euler4_t1.0','heun32_t1.0','C2 unmatched'),
    ('euler10_t1.0','heun32_t0.84','C3 matched'),('euler10_t1.0','heun32_t1.0','C3 unmatched')]
BRAKE,BRAKE_HOLD,SMOOTH,DEPART_HOLD=-3.0,0.3,0.5,1.0
BINARY=['hard_brake','lane_departure_any','wrong_lane'];CONTINUOUS=['max_decel','jerk_rms','lane_departure_frac']


async def ego_xy(path):
    t,xy=[],[]
    async for m in async_read_pb_log(path):
        if m.WhichOneof('log_entry')!='actor_poses':continue
        p=next((a for a in m.actor_poses.actor_poses if a.actor_id=='EGO'),None)
        if p is None:continue
        t.append(m.actor_poses.timestamp_us);xy.append((p.actor_pose.vec.x,p.actor_pose.vec.y))
    return np.array(t,dtype=np.float64)/1e6,np.array(xy)


def runs(mask,dt,hold):
    """True if mask has a run of consecutive True samples lasting at least hold seconds."""
    n=0
    for x in mask:
        n=n+1 if x else 0
        if n*dt>=hold:return True
    return False


def rollout_metrics(folder):
    d=pl.read_parquet(folder/'metrics.parquet').with_columns(pl.col('timestamps_us').cast(pl.Int64))
    rel=d.filter((pl.col('name')=='eval_relevant')&pl.col('valid')&(pl.col('values')>0))['timestamps_us'].to_numpy().astype(np.float64)/1e6
    lane=d.filter((pl.col('name')=='min_distance_to_lane_boundary_m')&pl.col('valid')).sort('timestamps_us')
    lt=lane['timestamps_us'].to_numpy().astype(np.float64)/1e6;lv=lane['values'].to_numpy();keep=(lt>=rel.min())&(lt<=rel.max())
    out=0.0==lv[keep];ldt=float(np.median(np.diff(lt)))
    wrong=d.filter((pl.col('name')=='wrong_lane')&pl.col('valid'))['values'].max()
    t,xy=asyncio.run(ego_xy(str(folder/'rollout.asl')));o=np.argsort(t);t,xy=t[o],xy[o]
    t,u=np.unique(t,return_index=True);xy=xy[u];dt=float(np.median(np.diff(t)));w=max(1,int(round(SMOOTH/dt)))
    smooth=lambda x:np.convolve(x,np.ones(w)/w,mode='same')
    speed=smooth(np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(t));ts=t[1:]
    acc=smooth(np.gradient(speed,ts));jerk=np.gradient(acc,ts);sel=(ts>=rel.min())&(ts<=rel.max())
    acc,jerk=acc[sel][w:-w],jerk[sel][w:-w]  # drop the moving-average edges
    return {'hard_brake':bool(runs(acc<=BRAKE,dt,BRAKE_HOLD)),'max_decel':float(-acc.min()),'jerk_rms':float(np.sqrt(np.mean(jerk**2))),
        'lane_departure_frac':float(out.mean()),'lane_departure_any':bool(runs(out,ldt,DEPART_HOLD)),'wrong_lane':bool((wrong or 0)>0),'pose_dt':dt}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--boot',type=int,default=2000);args=p.parse_args();tables={}
    for run in sorted(args.grid.glob('*_t*')):
        if '.failed' in run.name or not run.is_dir():continue
        tables[run.name]={Path(f).parent.parent.name:rollout_metrics(Path(f).parent) for f in glob.glob(str(run/'rollouts'/'*'/'*'/'metrics.parquet'))}
        print(run.name,len(tables[run.name]),flush=True)
    scenes=sorted(set.intersection(*[set(t) for t in tables.values()]));rng=np.random.default_rng(20261028)
    draws=rng.integers(0,len(scenes),(args.boot,len(scenes)));res={'scenes':len(scenes),'thresholds':{'brake_mps2':BRAKE,'brake_hold_s':BRAKE_HOLD,'smooth_s':SMOOTH,'depart_hold_s':DEPART_HOLD},
        'means':{c:{m:float(np.mean([tables[c][s][m] for s in scenes])) for m in BINARY+CONTINUOUS+['pose_dt']} for c in tables},'tests':[]}
    for a,b,label in PAIRS:
        for m in BINARY:
            x=np.array([tables[a][s][m] for s in scenes]);y=np.array([tables[b][s][m] for s in scenes]);oa,ob=int((x&~y).sum()),int((~x&y).sum())
            res['tests'].append({'label':label,'a':a,'b':b,'metric':m,'rate_a':float(x.mean()),'rate_b':float(y.mean()),'only_a':oa,'only_b':ob,
                'p_exact_mcnemar':float(binomtest(oa,oa+ob).pvalue) if oa+ob else 1.0})
        for m in CONTINUOUS:
            diff=np.array([tables[a][s][m]-tables[b][s][m] for s in scenes]);boot=diff[draws].mean(1)
            res['tests'].append({'label':label,'a':a,'b':b,'metric':m,'mean_a':float(np.mean([tables[a][s][m] for s in scenes])),
                'mean_b':float(np.mean([tables[b][s][m] for s in scenes])),'diff':float(diff.mean()),'diff_ci95':np.quantile(boot,[.025,.975]).tolist()})
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(res,indent=1))
    (args.out/'per_rollout.json').write_text(json.dumps(tables));(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for t in res['tests']:
        if 'p_exact_mcnemar' in t:print(f"{t['label']:13s} {t['metric']:19s} {t['rate_a']:.3f} vs {t['rate_b']:.3f} | {t['only_a']}:{t['only_b']} p={t['p_exact_mcnemar']:.3f}")
        else:print(f"{t['label']:13s} {t['metric']:19s} {t['mean_a']:.3f} vs {t['mean_b']:.3f} | diff {t['diff']:+.3f} [{t['diff_ci95'][0]:+.3f},{t['diff_ci95'][1]:+.3f}]")


if __name__=='__main__':main()
