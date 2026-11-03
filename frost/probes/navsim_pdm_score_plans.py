"""Score every trajectory of a navsim_temperature_probe run with the official NAVSIM PDM scorer (CPU, multiprocessing).

Calls navsim.evaluate.pdm_score.pdm_score (pdm_score.py:83) unchanged, with the PDMSimulator/PDMScorer built from
config/pdm_scoring/default_scoring_parameters.yaml of the chosen repo (the same parameters run_pdm_score.py instantiates).
The metric cache is read from $NAVSIM_EXP_ROOT/metric_cache/<log>/unknown/<token>/metric_cache.pkl (globbed, so a cache that is
still being generated works; MetricCacheLoader needs its final metadata csv). Tokens without a cache file are listed as missing.

  source configs/navsim.env; PYTHONPATH=.:$DIFFUSIONDRIVE_SRC python -m frost.probes.navsim_pdm_score_plans --plans DIR [--workers 24]
Output: DIR/pdm_scores.parquet (pdm_scores.json if pyarrow is unavailable) with token, seed, config, valid, pdms, nc, dac, ep, ttc, comfort, ddc;
the human plan is scored as config 'human', seed -1. --resume keeps rows already scored.
"""
import argparse
import hashlib
import json
import lzma
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np

ARTIFACTS = Path(os.environ.get('FROST_ARTIFACTS', 'artifacts'))
_G = {}


def cache_path(root, token, log=None):
    if log:
        p = root / log / 'unknown' / token / 'metric_cache.pkl'
        if p.exists():return p
    hits = list(root.glob(f'*/unknown/{token}/metric_cache.pkl'))
    return hits[0] if hits else None


def init_worker(repo):
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, repo)
    from omegaconf import OmegaConf
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    cfg = OmegaConf.load(Path(repo) / 'navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')
    sampling = TrajectorySampling(num_poses=cfg.proposal_sampling.num_poses, interval_length=cfg.proposal_sampling.interval_length)
    c = {k: v for k, v in cfg.scorer.config.items() if not k.startswith('_')}
    _G.update(simulator=PDMSimulator(sampling), scorer=PDMScorer(sampling, PDMScorerConfig(**c)), sampling=sampling)


def score_token(job):
    from dataclasses import asdict
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    token, path, items = job;rows = []
    try:
        with lzma.open(path, 'rb') as f:cache = pickle.load(f)
    except Exception as e:
        return [dict(token=t, seed=s, config=c, valid=False, error=f'cache: {e!r}') for s, c, _ in items]
    for seed, cfg, poses in items:
        row = dict(token=token, seed=seed, config=cfg, valid=True)
        try:
            r = asdict(pdm_score(metric_cache=cache, model_trajectory=Trajectory(np.asarray(poses, np.float32)), future_sampling=_G['sampling'],
                simulator=_G['simulator'], scorer=_G['scorer']))
            row.update({k: float(v) for k, v in r.items()})
        except Exception as e:
            row.update(valid=False, error=repr(e))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plans', type=Path, required=True, help='run directory with plans.npz');p.add_argument('--workers', type=int, default=16)
    p.add_argument('--repo', type=Path, default=Path(os.environ.get('DIFFUSIONDRIVE_SRC', 'third_party/diffusiondrive')));p.add_argument('--metric-cache', type=Path, default=None)
    p.add_argument('--scenes', type=Path, default=ARTIFACTS / 'navsim_scenes_v1_20261028.json');p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    import pandas as pd
    mc = args.metric_cache or Path(os.environ['NAVSIM_EXP_ROOT']) / 'metric_cache'
    logs = {s['token']: s['log'] for s in json.loads(args.scenes.read_text())['scenes']}
    with np.load(args.plans / 'plans.npz') as z:plans = {k: z[k] for k in z.files}
    by_token = {}
    for k, v in plans.items():
        token = k[:16]
        if k.endswith('_human'):seed, cfg = -1, 'human'
        else:_, rest = k.split('_s', 1);seed, cfg = int(rest.split('_', 1)[0]), rest.split('_', 1)[1]
        by_token.setdefault(token, []).append((seed, cfg, v))
    out = args.plans / 'pdm_scores.parquet';old = None
    if args.resume and out.exists():
        old = pd.read_parquet(out);have = set(zip(old.token, old.seed, old.config))
        by_token = {t: [i for i in its if (t, i[0], i[1]) not in have] for t, its in by_token.items()}
        by_token = {t: its for t, its in by_token.items() if its}
    jobs, missing = [], []
    for t, its in sorted(by_token.items()):
        path = cache_path(mc, t, logs.get(t))
        if path is None:missing.append(t)
        else:jobs.append((t, path, its))
    print(json.dumps({'tokens': len(by_token), 'to_score': len(jobs), 'missing_metric_cache': len(missing), 'plans': sum(len(j[2]) for j in jobs)}), flush=True)
    t0 = time.time();rows = []
    with mp.get_context('fork').Pool(min(args.workers, max(1, len(jobs))), init_worker, (str(args.repo),)) as pool:
        for i, r in enumerate(pool.imap_unordered(score_token, jobs)):
            rows += r
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):print(json.dumps({'tokens_done': i + 1, 'seconds': round(time.time() - t0)}), flush=True)
    df = pd.DataFrame(rows).rename(columns={'no_at_fault_collisions': 'nc', 'drivable_area_compliance': 'dac', 'ego_progress': 'ep',
        'time_to_collision_within_bound': 'ttc', 'comfortable': 'comfort', 'driving_direction_compliance': 'ddc', 'score': 'pdms'})
    if old is not None:df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values(['token', 'seed', 'config']).reset_index(drop=True)
    try:df.to_parquet(out, index=False)
    except Exception:df.to_json(args.plans / 'pdm_scores.json', orient='records')
    (args.plans / 'pdm_scores_manifest.json').write_text(json.dumps({'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'repo': str(args.repo), 'metric_cache': str(mc), 'missing_metric_cache_tokens': missing, 'rows': len(df), 'invalid': int((~df.valid).sum()),
        'seconds': time.time() - t0}, indent=1))
    ok = df[df.valid & (df.config != 'human')]
    print(json.dumps({'rows': len(df), 'invalid': int((~df.valid).sum()), 'mean_pdms_by_config': ok.groupby('config').pdms.mean().round(4).to_dict(),
        'human_pdms': df[df.config == 'human'].pdms.mean() if (df.config == 'human').any() else None}))


if __name__ == '__main__':main()
