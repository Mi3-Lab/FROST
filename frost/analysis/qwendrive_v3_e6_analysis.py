"""Euler-6 of Qwen-Drive-1.0 on the v3 held-out probe (protocols/qwen_drive.md, addendum 2).

Wrapper around frost.analysis.qwendrive_v3_analysis and the R2 check of frost.analysis.qwendrive_v3_heun_analysis (both left unchanged).
The E6 plans come from frost.probes.qwendrive_heun_control_v3 run with --configs euler:6:1,euler:10:1, so R2 compares the
regenerated reasoning traces and Euler-10 plans with the stored v3 ones (Heun-32 is not regenerated). If R2 passes, E6
joins the stored v3 tables as one more sampler (stored curve, stored Heun-32 tau=1 reference, same 2000 resamples with
seed 20261102). If R2 fails, the registered action is to stop and rerun with a same-run Heun-32; this script then only
writes the R2 result.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis import qwendrive_v3_analysis as v3
from frost.analysis import qwendrive_v3_heun_analysis as ha

E6, E4, E10 = 'euler6_t1.0', 'euler4_t1.0', 'euler10_t1.0'


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--e6', type=Path, nargs='+', required=True); p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--boot', type=int, default=2000)
    args = p.parse_args()
    records = {r['name']: r for r in json.loads(args.records.read_text()) if 'error' not in r}
    stored, _ = v3.load(args.probe, records); new, _ = v3.load(args.e6, records)
    traces = lambda dirs: {k: v for d in dirs for k, v in json.loads((d / 'reasoning.json').read_text()).items()}
    names_all = sorted(k.rsplit('_human', 1)[0] for k in stored if k.endswith('_human'))
    seeds = sorted({int(k.split('_s')[1].split('_')[0]) for k in stored if not k.endswith('_human')})
    missing = [n for n in names_all if f'{n}_human' not in new]
    ha.RECHECK = [E10]  # addendum 2: only E10 is regenerated
    check = ha.r2(stored, new, [n for n in names_all if n not in missing], seeds, records, traces(args.probe), traces(args.e6))
    check['missing_in_e6_run'] = missing
    result = {'protocol': 'protocols/qwen_drive.md, addendum 2', 'R2': check, 'boot': args.boot, 'seed': v3.SEED, 'margins': v3.MARGINS}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('R2', json.dumps(check)[:800])
    if not check['pass'] or missing:
        (args.out / 'summary.json').write_text(json.dumps(result, indent=2)); print('R2 failed or clips missing: stop (addendum 2)'); return
    plans = dict(stored); plans.update({k: v for k, v in new.items() if k.endswith('_' + E6)})
    tables = {}; excluded = {}
    for h, length in v3.HORIZONS.items():
        _, _, tables[h], excluded[h] = v3.clip_table(plans, records, length)
    names = sorted(set.intersection(*[set(tables[h]) for h in v3.HORIZONS]))
    rng = np.random.default_rng(v3.SEED); draws = [rng.integers(0, len(names), len(names)) for _ in range(args.boot)]
    result.update({'clips': len(names), 'distinct_clip_ids': len({records[n]['clip_id'] for n in names}), 'excluded': excluded, 'seeds': seeds,
                   'flip_reference': 'stored v3 heun32_t1.0', 'horizons': {}})
    for h in v3.HORIZONS:
        t = tables[h]; a = v3.arrays(t, names); idx = np.arange(len(names)); entry = {'samplers': {}, 'paired': {}}
        runs = {s: (v3.matched(a, idx, s), [v3.matched(a, x, s) for x in draws]) for s in (E4, E6, E10)}
        for s in (E4, E6, E10):
            e = v3.summarize(*runs[s], t, names, s)
            for m in v3.MARGINS:
                lo, hi = e[m]['residual_ci90']; e[m]['equivalent_table'] = bool(-v3.MARGINS[m] < lo and hi < v3.MARGINS[m])
            entry['samplers'][s] = e
        for x, y in ((E4, E6), (E6, E10)):
            for m in ('flip', 'ade1'):
                diff = [bx[m]['residual'] - by[m]['residual'] for bx, by in zip(runs[x][1], runs[y][1])]
                entry['paired'][f'{x}_minus_{y}_{m}'] = {'point': runs[x][0][m]['residual'] - runs[y][0][m]['residual'], 'ci95': v3.q(diff, [.025, .975]), 'ci90': v3.q(diff, [.05, .95])}
        result['horizons'][h] = entry
    s5 = result['horizons']['native_5s']['samplers']; e6 = s5[E6]
    result['predictions'] = {
        'QE1_tau_between_e4_e10': bool(e6['in_range'] and s5[E4]['tau_eff'] < e6['tau_eff'] < s5[E10]['tau_eff']),
        'QE2_ade1_equivalent': e6['ade1']['equivalent'],
        'QE3_flip_resid_point_between_e10_e4': bool(s5[E10]['flip']['residual'] < e6['flip']['residual'] < s5[E4]['flip']['residual']),
        'QE4_flip_equivalent': e6['flip']['equivalent']}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('clips', result['clips'], 'distinct', result['distinct_clip_ids'])
    for h, e in result['horizons'].items():
        for s, v in e['samplers'].items():
            print(h, s, 'tau', round(v['tau_eff'], 3), [round(x, 3) for x in v['tau_eff_ci95']], 'inrange', v['in_range'], v['boot_in_range'], 'spread', round(v['spread'], 4))
            for m in v3.METRICS:
                print('    ', m, f"samp {v[m]['sampler']:.4f} eff {v[m]['effect']:+.4f} res {v[m]['residual']:+.4f} ci95 {[round(x, 4) for x in v[m]['residual_ci95']]} ci90 {[round(x, 4) for x in v[m]['residual_ci90']]} eq={v[m].get('equivalent')} eqT={v[m].get('equivalent_table')}",
                      {k: v[m][k] for k in ('clips_with_flip_sampler', 'clips_with_flip_curve_bracket') if k in v[m]})
        print(h, 'paired', json.dumps(e['paired']))
    print('predictions', json.dumps(result['predictions']))


if __name__ == '__main__': main()
