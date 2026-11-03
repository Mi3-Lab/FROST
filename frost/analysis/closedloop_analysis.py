"""Pre-registered closed-loop analysis (protocols/closed_loop_alpasim.md).

Reads per-rollout AlpaSim metrics (long format: name, values, valid) for every configuration,
keeps the scenes completed by all configurations, and reports per-scene incident rates with
exact McNemar tests (scenes are the unit; one rollout per scene) for C1-C3, plus scene-bootstrap
intervals for distance metrics. Run with the AlpaSim venv (polars).
"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import binomtest

PAIRS=[('heun32_t0.65','heun32_t1.0','C1'),('euler4_t1.0','heun32_t0.65','C2 matched'),('euler4_t1.0','heun32_t1.0','C2 unmatched'),
    ('euler10_t1.0','heun32_t0.84','C3 matched'),('euler10_t1.0','heun32_t1.0','C3 unmatched')]


def scene_table(run):
    files=glob.glob(str(run/'rollouts'/'*'/'*'/'metrics.parquet'))
    if not files:return None
    d=pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col('valid'))
    m=d.group_by(['clipgt_id','name']).agg(pl.col('values').max()).pivot(on='name',index='clipgt_id',values='values')
    out={}
    for row in m.iter_rows(named=True):
        collision=max(row.get('collision_front') or 0,row.get('collision_lateral') or 0)>0
        out[row['clipgt_id']]={'collision_at_fault':bool(collision),'offroad':bool((row.get('offroad') or 0)>0),
            'incident':bool(collision or (row.get('offroad') or 0)>0),'dist_traveled_m':row.get('dist_traveled_m'),
            'dist_to_gt_trajectory':row.get('dist_to_gt_trajectory')}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();tables={}
    for run in sorted(args.grid.glob('*_t*')):
        if '.failed' in run.name or not run.is_dir():continue
        t=scene_table(run)
        if t:tables[run.name]=t
    common=sorted(set.intersection(*(set(t) for t in tables.values())));rng=np.random.default_rng(20261026)
    result={'configs':{k:len(v) for k,v in tables.items()},'common_scenes':len(common),'rates':{},'tests':[]}
    for name,t in tables.items():
        result['rates'][name]={m:float(np.mean([t[s][m] for s in common])) for m in ['collision_at_fault','offroad','incident']}
        result['rates'][name]['dist_traveled_m']=float(np.mean([t[s]['dist_traveled_m'] for s in common]))
    for a,b,label in PAIRS:
        if a not in tables or b not in tables:continue
        for metric in ['incident','collision_at_fault','offroad']:
            only_a=sum(tables[a][s][metric] and not tables[b][s][metric] for s in common);only_b=sum(tables[b][s][metric] and not tables[a][s][metric] for s in common)
            diff=np.array([float(tables[a][s][metric])-float(tables[b][s][metric]) for s in common])
            boots=[diff[rng.integers(len(diff),size=len(diff))].mean() for _ in range(4000)]
            result['tests'].append({'label':label,'a':a,'b':b,'metric':metric,'rate_a':float(np.mean([tables[a][s][metric] for s in common])),
                'rate_b':float(np.mean([tables[b][s][metric] for s in common])),'only_a':int(only_a),'only_b':int(only_b),
                'p_exact_mcnemar':float(binomtest(min(only_a,only_b),only_a+only_b).pvalue) if only_a+only_b else 1.,
                'diff_ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))]})
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'configs':result['configs'],'common':len(common),'rates':result['rates']},indent=1))
    for t in result['tests']:print(f"{t['label']:13s} {t['metric']:18s} {t['a']} {t['rate_a']:.3f} vs {t['b']} {t['rate_b']:.3f} | discordant {t['only_a']}:{t['only_b']} p={t['p_exact_mcnemar']:.3f}")


if __name__=='__main__':main()
