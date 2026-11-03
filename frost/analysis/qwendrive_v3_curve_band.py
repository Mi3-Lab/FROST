"""Per-tau clip-bootstrap 95% band of the Qwen-Drive v3 reference curve (native 5 s), for paper Fig. 4.

Same construction as the Alpamayo bands (paper/figures/figstyle.curve, paired=True): per clip, the metric of
Heun-32 at tau minus the metric at tau=1 (same noise); band = 2.5/97.5% quantiles of the clip-resampled mean.
Same clip table as frost.analysis.qwendrive_v3_analysis (clip_table, native 5 s), 2000 resamples, seed 20261102 (the very
resamples of the registered analysis: rng.integers(0, n, n)).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis.qwendrive_analysis import TAUS, REFERENCE
from frost.analysis.qwendrive_v3_analysis import load, clip_table, SEED, HORIZONS


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--records', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--boot', type=int, default=2000); args = p.parse_args()
    records = {r['name']: r for r in json.loads(args.records.read_text()) if 'error' not in r}
    plans, _ = load(args.probe, records)
    names, seeds, table, excluded = clip_table(plans, records, HORIZONS['native_5s'])
    rng = np.random.default_rng(SEED); draws = [rng.integers(0, len(names), len(names)) for _ in range(args.boot)]
    out = {'clips': len(names), 'excluded': excluded, 'boot': args.boot, 'seed': SEED, 'taus': TAUS, 'paired_to': REFERENCE, 'curve': {}}
    for m in ['ade1', 'flip']:
        base = np.array([table[n][REFERENCE][m] for n in names]); out['curve'][m] = {}
        for t in TAUS:
            d = np.array([table[n][f'heun32_t{t}'][m] for n in names]) - base
            b = np.array([d[x].mean() for x in draws]); lo, hi = np.quantile(b, [.025, .975])
            out['curve'][m][str(t)] = {'point': float(d.mean()), 'lo': float(lo), 'hi': float(hi)}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / 'summary.json').write_text(json.dumps(out, indent=2))
    (args.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for m, c in out['curve'].items():
        print(m, {t: [round(v['point'], 4), round(v['lo'], 4), round(v['hi'], 4)] for t, v in c.items()})


if __name__ == '__main__': main()
