"""Heun-3/Heun-5 control of Qwen-Drive-1.0 on the v3 held-out probe (protocols/qwen_drive.md, addendum 1).

Wrapper around frost.analysis.qwendrive_v3_analysis (left unchanged). The Heun-k plans come from a separate run
(frost.probes.qwendrive_heun_control_v3) that also regenerates Euler-10 and the Heun-32 tau=1 reference, so check R2 compares
the regenerated plans and reasoning traces with the stored v3 ones. If R2 passes, Heun-k joins the stored v3 tables as
one more sampler (stored curve, stored reference, same 2000 resamples with seed 20261102). If R2 fails, Heun-k flips are
measured against the same-run Heun-32 tau=1 (the Alpamayo Heun-control convention); tau_eff and ADE1 stay on the stored
curve. Above-grid spreads: the registered verdict marks the TOST not interpretable; the table cell uses tau_eff
truncated at 1 and the residual against tau = 1, bold when the 90% interval leaves the margin ('equivalent_table').
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis import qwendrive_v3_analysis as v3
from frost.analysis.qwendrive_analysis import REFERENCE
from frost.core.decisions import decisions

HEUN = ['heun3_t1.0', 'heun5_t1.0']
RECHECK = ['euler10_t1.0', REFERENCE]


def r2(stored, new, names, seeds, records, stored_traces, new_traces):
    same_text = [stored_traces.get(n) == new_traces.get(n) for n in names]
    out = {'reasoning_identical': int(sum(same_text)), 'clips': len(names), 'reasoning_frac': float(np.mean(same_text))}
    for c in RECHECK:
        d, agree = [], []
        for n in names:
            speed = records[n]['speed_mps']
            for s in seeds:
                a, b = stored[f'{n}_s{s}_{c}'], new[f'{n}_s{s}_{c}']
                d.append(float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).max()))
                x, y = decisions(a, speed), decisions(b, speed); agree.append((x['longitudinal'], x['lateral']) == (y['longitudinal'], y['lateral']))
        d = np.array(d)
        out[c] = {'pairs': len(d), 'median_m': float(np.median(d)), 'p99_m': float(np.quantile(d, .99)), 'max_m': float(d.max()),
                  'class_agreement': float(np.mean(agree)), 'pass': bool(np.median(d) <= 1e-3 and np.mean(agree) >= .995)}
    out['pass'] = bool(out['reasoning_frac'] >= .99 and all(out[c]['pass'] for c in RECHECK))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--heun', type=Path, nargs='+', required=True); p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--boot', type=int, default=2000)
    args = p.parse_args()
    records = {r['name']: r for r in json.loads(args.records.read_text()) if 'error' not in r}
    stored, origin = v3.load(args.probe, records); new, _ = v3.load(args.heun, records)
    traces = lambda dirs: {k: v for d in dirs for k, v in json.loads((d / 'reasoning.json').read_text()).items()}
    names_all = sorted(k.rsplit('_human', 1)[0] for k in stored if k.endswith('_human'))
    seeds = sorted({int(k.split('_s')[1].split('_')[0]) for k in stored if not k.endswith('_human')})
    missing = [n for n in names_all if f'{n}_human' not in new]
    check = r2(stored, new, [n for n in names_all if n not in missing], seeds, records, traces(args.probe), traces(args.heun))
    check['missing_in_heun_run'] = missing
    plans = dict(stored)
    for k, v in new.items():
        if any(k.endswith('_' + c) for c in HEUN): plans[k] = v
    if not check['pass']:  # fallback: Heun-k flips against the same-run reference (registered in addendum 1)
        same_ref = {k: v for k, v in new.items() if k.endswith('_' + REFERENCE)}
    tables = {}; excluded = {}
    for h, length in v3.HORIZONS.items():
        names, _, tables[h], excluded[h] = v3.clip_table(plans, records, length)
        if not check['pass']:
            for n in names:
                speed = records[n]['speed_mps']; cut = lambda a: a[:length]
                for c in HEUN:
                    ref = [decisions(cut(same_ref[f'{n}_s{s}_{REFERENCE}']), speed) for s in seeds]
                    dec = [decisions(cut(plans[f'{n}_s{s}_{c}']), speed) for s in seeds]
                    lon = [d['longitudinal'] != r['longitudinal'] for d, r in zip(dec, ref)]; lat = [d['lateral'] != r['lateral'] for d, r in zip(dec, ref)]
                    tables[h][n][c].update({'flip': float(np.mean([a or b for a, b in zip(lon, lat)])), 'flip_long': float(np.mean(lon)), 'flip_lat': float(np.mean(lat))})
    names = sorted(set.intersection(*[set(tables[h]) for h in v3.HORIZONS]))
    rng = np.random.default_rng(v3.SEED); draws = [rng.integers(0, len(names), len(names)) for _ in range(args.boot)]
    result = {'protocol': 'protocols/qwen_drive.md, addendum 1', 'clips': len(names), 'distinct_clip_ids': len({records[n]['clip_id'] for n in names}),
              'excluded': excluded, 'seeds': seeds, 'boot': args.boot, 'seed': v3.SEED, 'margins': v3.MARGINS, 'R2': check,
              'flip_reference': 'stored v3 heun32_t1.0' if check['pass'] else 'same-run heun32_t1.0 (R2 failed)', 'horizons': {}}
    for h in v3.HORIZONS:
        t = tables[h]; a = v3.arrays(t, names); idx = np.arange(len(names)); entry = {'samplers': {}, 'paired': {}}
        runs = {s: (v3.matched(a, idx, s), [v3.matched(a, x, s) for x in draws]) for s in HEUN + ['euler10_t1.0']}
        for s in HEUN + ['euler10_t1.0']:
            e = v3.summarize(*runs[s], t, names, s)
            for m in v3.MARGINS:  # table convention: bold rule on the 90% interval, regardless of the grid
                lo, hi = e[m]['residual_ci90']; e[m]['equivalent_table'] = bool(-v3.MARGINS[m] < lo and hi < v3.MARGINS[m])
            entry['samplers'][s] = e
        for x, y in (('heun5_t1.0', 'euler10_t1.0'), ('heun3_t1.0', 'heun5_t1.0')):
            for m in ('flip', 'ade1'):
                diff = [bx[m]['residual'] - by[m]['residual'] for bx, by in zip(runs[x][1], runs[y][1])]
                entry['paired'][f'{x}_minus_{y}_{m}'] = {'point': runs[x][0][m]['residual'] - runs[y][0][m]['residual'], 'ci95': v3.q(diff, [.025, .975]), 'ci90': v3.q(diff, [.05, .95])}
        result['horizons'][h] = entry
    e5 = result['horizons']['native_5s']; s5 = e5['samplers']
    result['predictions'] = {
        'QH1_heun5_tau_ge_0.85': bool(s5['heun5_t1.0']['tau_eff'] >= .85), 'QH1_heun3_tau_ge_0.85': bool(s5['heun3_t1.0']['tau_eff'] >= .85),
        'QH2_heun5_flip_resid_gt_e10': bool(e5['paired']['heun5_t1.0_minus_euler10_t1.0_flip']['ci95'][0] > 0),
        'QH3_heun5_flip_resid_point_in_0_2pt': bool(0 < s5['heun5_t1.0']['flip']['residual'] < .02),
        'QH3_table_not_bold': s5['heun5_t1.0']['flip']['equivalent_table'],
        'QH4_heun5_ade1_equivalent_registered': s5['heun5_t1.0']['ade1']['equivalent'], 'QH4_heun5_ade1_equivalent_table': s5['heun5_t1.0']['ade1']['equivalent_table'],
        'QH5_heun3_flip_resid_gt_heun5': bool(e5['paired']['heun3_t1.0_minus_heun5_t1.0_flip']['point'] > 0)}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    (args.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('clips', result['clips'], 'R2', json.dumps(check)[:800])
    for h, e in result['horizons'].items():
        for s, v in e['samplers'].items():
            print(h, s, 'tau', round(v['tau_eff'], 3), [round(x, 3) for x in v['tau_eff_ci95']], 'inrange', v['in_range'], v['boot_in_range'], 'spread', round(v['spread'], 4))
            for m in v3.METRICS:
                print('    ', m, f"samp {v[m]['sampler']:.4f} eff {v[m]['effect']:+.4f} res {v[m]['residual']:+.4f} ci95 {[round(x, 4) for x in v[m]['residual_ci95']]} ci90 {[round(x, 4) for x in v[m]['residual_ci90']]} eq={v[m].get('equivalent')} eqT={v[m].get('equivalent_table')}",
                      {k: v[m][k] for k in ('clips_with_flip_sampler', 'clips_with_flip_curve_bracket') if k in v[m]})
        print(h, 'paired', json.dumps(e['paired']))
    print('predictions', json.dumps(result['predictions']))


if __name__ == '__main__': main()
