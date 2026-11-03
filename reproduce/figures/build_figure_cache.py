"""Build results/figure_cache/ from the raw per-clip plans (needs $FROST_ARTIFACTS with the temperature probes
and context caches listed in figstyle.SAMPLES).

Writes
  per_clip_pooled.npz         spread, ADE1, minADE6 and flip share per clip for every model and configuration of the
                              400 held-out clips, computed by frost.analysis.paper_numbers (float64, exact values)
  plans_orig_alpamayo15.npz   the six-seed plans of the page-one example clip (Alpamayo-1.5, original test sample)
The figure scripts use these files only when the raw plans are not available.
"""
import argparse

import numpy as np

import figstyle as fs


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--clip', default='context_101');a = p.parse_args()
    fs.CACHE.mkdir(parents=True, exist_ok=True)
    fs.use('pooled');names, T = fs.compute_tables()
    arrays = {'names': np.array(names)}
    for model, table in T.items():
        for config, per in table.items():
            arrays[f'{model}/{config}'] = np.array([[per[n][m] for m in fs.CACHE_METRICS] for n in names], dtype=np.float64)
    np.savez_compressed(fs.CACHE / 'per_clip_pooled.npz', **arrays)
    fs.use('orig');P = fs.plans('alpamayo15')
    keep = {k: v for k, v in P.items() if k.startswith(f'{a.clip}_s') and k.endswith(('_heun32_t1.0', '_euler4_t1.0', '_euler1_t1.0'))}
    np.savez_compressed(fs.CACHE / 'plans_orig_alpamayo15.npz', **keep)
    print('wrote', fs.CACHE, len(arrays) - 1, 'per-clip tables,', len(keep), 'example plans')


if __name__ == '__main__':
    main()
