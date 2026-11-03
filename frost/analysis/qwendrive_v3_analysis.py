"""FROST of Qwen-Drive-1.0 (RL planner) on the v3 held-out probe (protocols/qwen_drive.md).

Same estimator as frost.analysis.qwendrive_analysis (addendum 12): clip is the unit; tau_eff is where the mean seed spread of the
Heun-32 curve on [0, 0.9] (tau 1..0.3) equals the sampler's; residuals are sampler minus the curve interpolated at that
spread; 2000 clip resamples (seed 20261102) re-solve tau_eff. Additions registered in the protocol: the 4 s common
horizon (first 40 points) next to the native 5 s, flips split into longitudinal and lateral, the TOST of the horizon
change in tau_eff (+-0.03), checks R0 (Heun-64 vs Heun-32 at tau 1) and R1 (lateral sign against the logged future).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis.qwendrive_analysis import TAUS, CURVE, REFERENCE
from frost.core.decisions import decisions, OFFSET_M

SAMPLERS = [f'euler{k}_t1.0' for k in [1, 2, 4, 10]]
METRICS = ['ade1', 'minade6', 'flip', 'flip_long', 'flip_lat']
MARGINS = {'ade1': .05, 'flip': .02, 'flip_long': .02, 'flip_lat': .02}
TAU_MARGIN = .03
HORIZONS = {'native_5s': 50, 'common_4s': 40}
SEED = 20261102


def load(probes, records):
    plans, origin = {}, {}
    for probe in probes:
        with np.load(probe / 'plans.npz') as z:
            names = sorted({k.rsplit('_human', 1)[0] for k in z.files if k.endswith('_human')})
            for n in names:
                cid = records[n]['clip_id']
                if cid in origin: continue  # duplicate clip id: keep the first shard
                origin[cid] = probe.name
                for k in z.files:
                    if k == f'{n}_human' or k.startswith(f'{n}_s'): plans[k] = z[k]
    return plans, origin


def clip_table(plans, records, horizon):
    names = sorted(k.rsplit('_human', 1)[0] for k in plans if k.endswith('_human'))
    seeds = sorted({int(k.split('_s')[1].split('_')[0]) for k in plans if not k.endswith('_human')})
    configs = sorted({k.split('_', 3)[3] for k in plans if not k.endswith('_human')})
    table, excluded = {}, []
    for n in names:
        keys = [f'{n}_s{s}_{c}' for s in seeds for c in configs]
        if any(k not in plans or not np.isfinite(plans[k]).all() for k in keys) or not np.isfinite(plans[f'{n}_human']).all():
            excluded.append(n); continue
        cut = lambda a: a[:horizon]
        human = cut(plans[f'{n}_human'])[:, :2]; speed = records[n]['speed_mps']
        dec = {c: [decisions(cut(plans[f'{n}_s{s}_{c}']), speed) for s in seeds] for c in configs}
        ref = dec[REFERENCE]; table[n] = {}
        for c in configs:
            trajs = [cut(plans[f'{n}_s{s}_{c}'])[:, :2] for s in seeds]; err = [np.linalg.norm(t - human, axis=1).mean() for t in trajs]
            lon = [d['longitudinal'] != r['longitudinal'] for d, r in zip(dec[c], ref)]
            lat = [d['lateral'] != r['lateral'] for d, r in zip(dec[c], ref)]
            table[n][c] = {'spread': float(np.mean([np.linalg.norm(a - b, axis=1).mean() for i, a in enumerate(trajs) for b in trajs[i + 1:]])),
                           'ade1': float(np.mean(err)), 'minade6': float(min(err)),
                           'flip': float(np.mean([a or b for a, b in zip(lon, lat)])), 'flip_long': float(np.mean(lon)), 'flip_lat': float(np.mean(lat))}
    return names if not excluded else [n for n in names if n not in excluded], seeds, table, excluded


def arrays(table, names):
    configs = table[names[0]].keys()
    return {c: {m: np.array([table[n][c][m] for n in names]) for m in ['spread'] + METRICS} for c in configs}


def matched(arr, idx, sampler):
    """Same interpolation as frost.analysis.qwendrive_analysis.matched, over the extended metric list; idx indexes clips."""
    mean = lambda c, m: float(arr[c][m][idx].mean())
    spreads = np.array([mean(c, 'spread') for c in CURVE]); o = np.argsort(spreads); s = mean(sampler, 'spread')
    out = {'tau_eff': float(np.interp(s, spreads[o], np.array(TAUS)[o])), 'in_range': bool(spreads.min() <= s <= spreads.max()), 'spread': s,
           'below_grid': bool(s < spreads.min()), 'above_grid': bool(s > spreads.max())}
    for m in METRICS:
        at = float(np.interp(s, spreads[o], np.array([mean(c, m) for c in CURVE])[o])); v = mean(sampler, m)
        out[m] = {'reference': mean(REFERENCE, m), 'sampler': v, 'matched': at, 'effect': v - mean(REFERENCE, m), 'residual': v - at}
    out['cooling_ade1'] = out['ade1']['matched'] - out['ade1']['reference']  # curve ADE1 at tau_eff minus tau=1
    return out


def q(x, p): return np.quantile(np.asarray(x, float), p).tolist()


def summarize(point, boots, table, names, sampler):
    frac = float(np.mean([b['in_range'] for b in boots]))
    e = {'tau_eff': point['tau_eff'], 'tau_eff_ci95': q([b['tau_eff'] for b in boots], [.025, .975]),
         'tau_eff_ci90': q([b['tau_eff'] for b in boots], [.05, .95]), 'in_range': point['in_range'], 'below_grid': point['below_grid'],
         'above_grid': point['above_grid'], 'boot_in_range': frac, 'spread': point['spread'],
         'interpretable': bool(point['in_range'] and frac >= .95)}
    for m in METRICS:
        res = [b[m]['residual'] for b in boots]; eff = [b[m]['effect'] for b in boots]
        e[m] = {**point[m], 'residual_ci95': q(res, [.025, .975]), 'residual_ci90': q(res, [.05, .95]), 'effect_ci95': q(eff, [.025, .975])}
        if m in MARGINS:
            lo, hi = e[m]['residual_ci90']
            e[m]['equivalent'] = bool(e['interpretable'] and -MARGINS[m] < lo and hi < MARGINS[m])
        if m.startswith('flip'):
            # distinct clips behind the effect: any flip in the sampler, any flip in the two curve points bracketing tau_eff
            lo_t = max([t for t in TAUS if t <= point['tau_eff']], default=TAUS[-1]); hi_t = min([t for t in TAUS if t >= point['tau_eff']], default=1.)
            e[m]['clips_with_flip_sampler'] = int(sum(table[n][sampler][m] > 0 for n in names))
            e[m]['clips_with_flip_curve_bracket'] = {str(t): int(sum(table[n][f'heun32_t{t}'][m] > 0 for n in names)) for t in sorted({lo_t, hi_t})}
    cool = [b['cooling_ade1'] for b in boots]
    e['cooling_ade1'] = {'point': point['cooling_ade1'], 'ci95': q(cool, [.025, .975]), 'ci90': q(cool, [.05, .95])}
    e['cooling_ade1']['equivalent'] = bool(e['interpretable'] and -MARGINS['ade1'] < e['cooling_ade1']['ci90'][0] and e['cooling_ade1']['ci90'][1] < MARGINS['ade1'])
    return e


def checks(plans, names, seeds, records):
    d, agree = [], []
    for n in names:
        speed = records[n]['speed_mps']
        for s in seeds:
            a, b = plans[f'{n}_s{s}_heun64_t1.0'], plans[f'{n}_s{s}_{REFERENCE}']
            d.append(float(np.abs(a[:, :2] - b[:, :2]).max()))
            x, y = decisions(a, speed), decisions(b, speed); agree.append((x['longitudinal'], x['lateral']) == (y['longitudinal'], y['lateral']))
    r0 = {'pairs': len(d), 'max_abs_m': max(d), 'median_m': float(np.median(d)), 'class_agreement': float(np.mean(agree))}
    r0['pass'] = bool(r0['max_abs_m'] <= .01 and r0['class_agreement'] >= .995)
    sign = []
    for n in names:
        y_h = plans[f'{n}_human'][39, 1]
        if abs(y_h) > OFFSET_M:
            y_p = np.mean([plans[f'{n}_s{s}_euler10_t1.0'][39, 1] for s in seeds]); sign.append(np.sign(y_p) == np.sign(y_h))
    r1 = {'clips': len(sign), 'agreement': float(np.mean(sign)) if sign else None}
    r1['pass'] = bool(sign and r1['agreement'] >= .8)
    return {'R0': r0, 'R1': r1}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--records', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--boot', type=int, default=2000)
    args = p.parse_args()
    records = {r['name']: r for r in json.loads(args.records.read_text()) if 'error' not in r}
    plans, origin = load(args.probe, records)
    tables = {}; excluded = {}
    for h, length in HORIZONS.items():
        names, seeds, tables[h], excluded[h] = clip_table(plans, records, length)
    names = sorted(set.intersection(*[set(tables[h]) for h in HORIZONS]))
    rng = np.random.default_rng(SEED); draws = [rng.integers(0, len(names), len(names)) for _ in range(args.boot)]
    result = {'protocol': 'protocols/qwen_drive.md', 'clips': len(names), 'distinct_clip_ids': len({records[n]['clip_id'] for n in names}),
              'shards': {k: sum(v == k for v in origin.values()) for k in sorted(set(origin.values()))}, 'excluded': excluded, 'seeds': seeds,
              'boot': args.boot, 'seed': SEED, 'margins': MARGINS, 'tau_margin': TAU_MARGIN, 'checks': checks(plans, names, seeds, records), 'horizons': {}}
    runs = {}
    for h in HORIZONS:
        t = tables[h]
        entry = {'curve': {c: {m: float(np.mean([t[n][c][m] for n in names])) for m in ['spread', 'ade1', 'minade6', 'flip', 'flip_long', 'flip_lat']} for c in CURVE},
                 'samplers': {}}
        for s in SAMPLERS:
            a = arrays(t, names); point = matched(a, np.arange(len(names)), s); boots = [matched(a, x, s) for x in draws]; runs[(h, s)] = (point, boots)
            entry['samplers'][s] = summarize(point, boots, t, names, s)
        result['horizons'][h] = entry
    result['delta_tau_4s_minus_5s'] = {}
    for s in SAMPLERS:
        (p5, b5), (p4, b4) = runs[('native_5s', s)], runs[('common_4s', s)]
        diff = [y['tau_eff'] - x['tau_eff'] for x, y in zip(b5, b4)]
        ok = all(result['horizons'][h]['samplers'][s]['interpretable'] for h in HORIZONS)
        ci90 = q(diff, [.05, .95])
        result['delta_tau_4s_minus_5s'][s] = {'point': p4['tau_eff'] - p5['tau_eff'], 'ci95': q(diff, [.025, .975]), 'ci90': ci90,
                                              'equivalent': bool(ok and -TAU_MARGIN < ci90[0] and ci90[1] < TAU_MARGIN)}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    (args.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('clips', result['clips'], 'checks', json.dumps(result['checks']))
    for h, e in result['horizons'].items():
        print(h, 'curve spread', {c: round(v['spread'], 4) for c, v in e['curve'].items()})
        for s, v in e['samplers'].items():
            print(' ', s, 'tau', round(v['tau_eff'], 3), [round(x, 3) for x in v['tau_eff_ci95']], 'inrange', v['in_range'], v['boot_in_range'])
            for m in METRICS:
                print('    ', m, f"eff {v[m]['effect']:+.4f} res {v[m]['residual']:+.4f} ci95 {[round(x, 4) for x in v[m]['residual_ci95']]} ci90 {[round(x, 4) for x in v[m]['residual_ci90']]} eq={v[m].get('equivalent')}")
    print('dtau', json.dumps(result['delta_tau_4s_minus_5s']))


if __name__ == '__main__': main()
