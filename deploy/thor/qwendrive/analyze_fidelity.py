"""Fidelity of a Thor Qwen-Drive run against the cluster PyTorch reference (qwendrive_v3 probe, euler10 tau 1, seeds 9201-9206).
Reports: reasoning text identical n/N; per-clip plan change (median over seeds of max_t |xy - xy_ref|, cm) as median/p90/max over clips,
all-clips and text-identical clips only; and the seed-spread ratio (mean distance of the 6 seed plans to their mean, run / reference,
with a bootstrap CI over clips). Usage: analyze_fidelity.py --run DIR --ref DIR [--n 40]"""
import argparse, json
from pathlib import Path
import numpy as np


def spread(P):  # P [6,T,2]
    return float(np.linalg.norm(P - P.mean(0, keepdims=True), axis=-1).mean())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--ref', type=Path, required=True)
    p.add_argument('--n', type=int, default=40); p.add_argument('--seed0', type=int, default=9201); p.add_argument('--out', type=Path)
    a = p.parse_args()
    run = np.load(a.run / 'plans.npz'); ref = np.load(a.ref / 'plans.npz')
    tr = json.loads((a.run / 'reasoning.json').read_text()); tf = json.loads((a.ref / 'reasoning.json').read_text())
    names = [f'context_{i:03d}' for i in range(a.n)]; names = [n for n in names if n in tr]
    seeds = [a.seed0 + i for i in range(6)]
    same = [tr[n].strip() == tf[n].strip() for n in names]
    d, sp_run, sp_ref = [], [], []
    for n in names:
        R = np.stack([run[f'{n}_s{s}'][:, :2] for s in seeds]); F = np.stack([ref[f'{n}_s{s}_euler10_t1.0'][:, :2] for s in seeds])
        d.append(float(np.median(np.linalg.norm(R - F, axis=-1).max(-1))) * 100); sp_run.append(spread(R)); sp_ref.append(spread(F))
    d = np.array(d); same = np.array(same); sr, sf = np.array(sp_run), np.array(sp_ref)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(names), (4000, len(names)))
    ratio = sr.mean() / sf.mean(); boots = sr[idx].mean(1) / sf[idx].mean(1)
    q = lambda x: [round(float(np.percentile(x, 50)), 1), round(float(np.percentile(x, 90)), 1), round(float(x.max()), 1)] if len(x) else None
    out = dict(n=len(names), text_identical=int(same.sum()), dplan_cm_median_p90_max=q(d), dplan_cm_text_identical=q(d[same]),
               spread_ratio=round(float(ratio), 4), spread_ci95=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
               spread_run_m=round(float(sr.mean()), 4), spread_ref_m=round(float(sf.mean()), 4), differing_clips=[n for n, s in zip(names, same) if not s])
    print(json.dumps(out, indent=1))
    if a.out: a.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
