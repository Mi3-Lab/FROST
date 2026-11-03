"""Thor addendum 3 analysis (protocols/thor.md, Addendum 3, registered 31/10/2026 23:37 PDT): tau_eff with the compiled
VLM prefix, predictions W1-W6.

Thin wrapper only. Every statistic comes from frost.analysis.thor_analysis, imported unchanged (tables, arrays, matched, ci, tost,
plan_diffs, load_probe, the config lists and TAU_MARGIN) and from frost.core.matched_temperature_stats (summarize, MARGINS).
The wrapper only points those functions at the addendum-3 directories and generalizes the reference of the paired shift
(addendum 3 compares against V1 and V3, not against P0). The resampling mirrors thor_analysis.main exactly: rng seeded once,
index 0 = the full sample, then `boot` clip resamples drawn with rng.integers, shared by every variant (clips resampled
jointly), tau_eff re-solved in every resample. Boot seed 20261031, as registered in addendum 2 (addendum 3: sample, seeds and
configs identical to addendum 2). With these resamples, the V0/V1 numbers must equal the addendum-2 summary (checked below).

Variants: V0 = trt_bf16 and V1 = trt_fp8 (addendum 2, PyTorch bf16 prefix); V3 = comp_fp8, V4 = comp_fp8pre_awqfulldec
(compiled VLM prefix, same FP8 field as V1).

Usage: python -m frost.analysis.thor_addendum3_analysis --out artifacts/thor_addendum3_analysis_v1_20261101
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis import thor_analysis as ta
from frost.analysis.thor_analysis import (TAUS, CURVE, REFERENCE, SAMPLERS, CONFIGS, TAU_MARGIN, tables, arrays, matched, ci, tost,
                                 plan_diffs, load_probe)
from frost.core.matched_temperature_stats import METRICS, MARGINS, summarize

BOOT_SEED = 20261031
E10, E4 = 'euler10_t1.0', 'euler4_t1.0'
TRT = Path('artifacts/thor_trt_frost400_20261031'); VLM = Path('artifacts/thor_vlm_frost400_20261101')
VARIANTS = {'V0': TRT / 'trt_bf16', 'V1': TRT / 'trt_fp8', 'V3': VLM / 'comp_fp8', 'V4': VLM / 'comp_fp8pre_awqfulldec'}
L = 3456; MAX_NEW = 256


def timing(path):
    out = {}
    for line in (path / 'timing.jsonl').read_text().splitlines():
        r = json.loads(line); out[r['clip']] = r  # last record per clip wins (resume)
    return out


def cot(text):
    return text.split('<|cot_end|>')[0].strip()


def shift(runs, arr, v, ref, samples):
    """thor_analysis.main step (3), with `ref` in place of P0."""
    sh = {}
    for s in SAMPLERS:
        d = np.array([x['tau_eff'] - y['tau_eff'] for x, y in zip(runs[v][s], runs[ref][s])])
        sh[s] = {'dtau_eff': float(d[0]), 'ci95': ci(d[1:]), **tost(d[1:], TAU_MARGIN)}
    nat = [matched({**arr[ref], 'native': arr[v][REFERENCE]}, i, 'native') for i in samples]
    sh['heun32_native_tau'] = {'tau_on_ref_curve': nat[0]['tau_eff'], 'ci95': ci([x['tau_eff'] for x in nat[1:]]), 'in_range': nat[0]['in_range'],
                               'dtau': nat[0]['tau_eff'] - 1., 'spread_ratio_vs_ref': nat[0]['spread'] / float(arr[ref][REFERENCE]['spread'].mean())}
    return sh


def tau_verdict(sh):
    parts = {s: sh[s]['equivalent'] for s in [E10, E4]}
    return {'parts': parts, 'verdict': 'PASS' if all(parts.values()) else 'FAIL',
            'falsified': [s for s in parts if not parts[s] and (sh[s]['ci95'][0] > 0 or sh[s]['ci95'][1] < 0) and abs(sh[s]['dtau_eff']) > TAU_MARGIN],
            'inconclusive': [s for s in parts if not parts[s] and sh[s]['ci95'][0] <= 0 <= sh[s]['ci95'][1]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--range', type=int, nargs=2, default=[0, 400]); p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--addendum2', type=Path, default=Path('artifacts/thor_trt_frost400_analysis_v0v1v2_v1_20261031/summary.json'))
    p.add_argument('--pytorch-text', type=Path, default=Path('artifacts/thor_v1_20261030/P1/reasoning.jsonl'),
                   help='PyTorch bf16 VLM reasoning on the Thor (P1 runs the bf16 VLM); the only local per-clip PyTorch text')
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    records = {r['name']: r for r in json.loads((a.data / 'records.json').read_text()) if 'error' not in r}
    want = [f'context_{i:03d}' for i in range(*a.range)]
    probes = {v: load_probe(d) for v, d in VARIANTS.items()}
    seeds = probes['V0'][1]; assert all(pr[1] == seeds for pr in probes.values())
    has = lambda plans, n: all(f'{n}_s{s}_{c}' in plans for s in seeds for c in CONFIGS)
    names = [n for n in want if all(has(pr[0], n) for pr in probes.values())]
    res = {'protocol': 'protocols/thor.md Addendum 3 (31/10/2026 23:37 PDT)', 'code': 'frost.analysis.thor_analysis functions unchanged, via frost.analysis.thor_addendum3_analysis',
           'range': a.range, 'boot': a.boot, 'boot_seed': BOOT_SEED, 'margins': {**MARGINS, 'tau_eff': TAU_MARGIN},
           'variants': {v: str(d) for v, d in VARIANTS.items()}, 'clips_requested': len(want), 'clips_used': len(names),
           'probes': {v: {'complete': pr[2], 'clips': sum(has(pr[0], n) for n in want)} for v, pr in probes.items()}}
    rng = np.random.default_rng(BOOT_SEED)
    samples = [np.arange(len(names))] + [rng.integers(0, len(names), len(names)) for _ in range(a.boot)]

    # Matched-temperature test per variant against its own Heun-32 curve (thor_analysis.main step (2)).
    arr, runs, res['variants_own_curve'] = {}, {}, {}
    for v in VARIANTS:
        table = tables(probes[v][0], names, seeds, a.data, records); arr[v] = arrays(table, names)
        runs[v] = {s: [matched(arr[v], i, s) for i in samples] for s in SAMPLERS + [REFERENCE]}
        for s in SAMPLERS:
            ref = summarize(table, names, CURVE, TAUS, s); pt = runs[v][s][0]
            assert abs(ref['tau_eff'] - pt['tau_eff']) < 1e-9 and all(abs(ref[m]['residual'] - pt[m]['residual']) < 1e-9 for m in METRICS), (v, s)
        entry = {}
        for s in SAMPLERS:
            r = runs[v][s]; pt = r[0]; bt = r[1:]; e = {'tau_eff': [pt['tau_eff'], *ci([b['tau_eff'] for b in bt])], 'in_range': pt['in_range'], 'spread': pt['spread']}
            for m in METRICS:
                rs = [b[m]['residual'] for b in bt]
                e[m] = {'value': pt[m]['value'], 'effect': [pt[m]['effect'], *ci([b[m]['effect'] for b in bt])], 'residual': [pt[m]['residual'], *ci(rs)], **tost(rs, MARGINS[m])}
            entry[s] = e
        res['variants_own_curve'][v] = entry

    # Reproduction check: V0/V1 must equal the addendum-2 analysis (same code, clips, seed, resamples).
    if a.addendum2.exists():
        a2 = json.loads(a.addendum2.read_text()); chk = {}
        for v, k in [('V0', 'P0'), ('V1', 'P1')]:
            for s in [E10, E4]:
                chk[f'{v}_{s}_tau_eff'] = abs(a2['precisions'][k][s]['tau_eff'][0] - res['variants_own_curve'][v][s]['tau_eff'][0]) < 1e-12 and \
                    np.allclose(a2['precisions'][k][s]['tau_eff'][1:], res['variants_own_curve'][v][s]['tau_eff'][1:], atol=1e-12)
        res['addendum2_reproduced'] = chk

    # Paired shifts (step (3) with a chosen reference).
    pairs = [('V3', 'V1'), ('V4', 'V1'), ('V4', 'V3'), ('V3', 'V0'), ('V4', 'V0'), ('V1', 'V0')]
    res['shift'] = {f'{v}-{r}': shift(runs, arr, v, r, samples) for v, r in pairs}
    if a.addendum2.exists():
        res['addendum2_reproduced']['V1-V0_E10_dtau'] = abs(a2['shift_vs_P0']['P1'][E10]['dtau_eff'] - res['shift']['V1-V0'][E10]['dtau_eff']) < 1e-12

    # W3: E10 plans vs V1, same clip and seed (plan_diffs restricted to E10 keys).
    res['plans_vs_V1_E10'] = {}
    for v in ['V3', 'V4']:
        b = {k: x for k, x in probes['V1'][0].items() if k.endswith('_' + E10)}
        per = plan_diffs(probes[v][0], b, names, seeds); med = np.array([float(np.median(per[n])) for n in names])
        allp = np.array([x for n in names for x in per[n]])
        res['plans_vs_V1_E10'][v] = {'median_of_clip_medians_m': float(np.median(med)), 'median_of_clip_medians_ci95': ci([np.median(med[i]) for i in samples[1:]]),
                                    'median_all_plans_m': float(np.median(allp)), 'p90_all_plans_m': float(np.quantile(allp, .9)), 'max_m': float(allp.max()),
                                    'plans_compared': int(allp.size), 'clips_with_median_gt_1m': int((med > 1).sum())}

    # W5: spread ratio V/V1 for Heun-32 tau=1 and E10. Primary: ratio of clip-mean spreads (the thor_analysis spread and
    # the definition of spread_ratio_vs_P0); sensitivity: geometric mean of per-clip ratios (analyze_vlm_val style).
    res['spread_ratio_vs_V1'] = {}
    for v in ['V3', 'V4']:
        e = {}
        for c in [REFERENCE, E10]:
            x, y = arr[v][c]['spread'], arr['V1'][c]['spread']
            rat = [float(x[i].mean() / y[i].mean()) for i in samples]; lr = np.log(x / y)
            e[c] = {'ratio_of_means': rat[0], 'ci95': ci(rat[1:]), 'geomean_per_clip': float(np.exp(lr.mean())),
                    'geomean_ci95': ci([float(np.exp(lr[i].mean())) for i in samples[1:]])}
        res['spread_ratio_vs_V1'][v] = e

    # W4 / W6: reasoning text and skipped clips from timing.jsonl.
    tm = {v: timing(VARIANTS[v]) for v in VARIANTS}
    pt_text = {}
    if a.pytorch_text.exists():
        for line in a.pytorch_text.read_text().splitlines():
            r = json.loads(line); pt_text[r['clip']] = r['extra']['cot'][0][0][0].strip()
    res['reasoning'] = {}
    for v in ['V3', 'V4']:
        common = [n for n in names if n in pt_text and n in tm[v]]
        same = [cot(tm[v][n]['text']) == pt_text[n] for n in common]
        pm = [tm[v][n]['P'] == tm['V1'][n]['P'] for n in names]
        res['reasoning'][v] = {'text_reference': str(a.pytorch_text), 'clips_with_pytorch_text': len(common), 'text_identical': int(sum(same)),
                               'fraction_identical': float(np.mean(same)) if common else None,
                               'mismatch_clips': [n for n, s in zip(common, same) if not s],
                               'P_equal_to_V1_pytorch_prefix_400': int(sum(pm)), 'fraction_P_equal_400': float(np.mean(pm)),
                               'P_mismatch_in_text_reference_clips': [n for n in common if tm[v][n]['P'] != tm['V1'][n]['P']],
                               'note': 'P equal is necessary for identical text, so fraction_P_equal_400 is an upper bound on the 400-clip text identity',
                               'text_identical_V3_vs_V4': int(sum(cot(tm['V3'][n]['text']) == cot(tm['V4'][n]['text']) for n in names)) if v == 'V4' else None,
                               'vlm_s_median': float(np.median([tm[v][n]['vlm_s'] for n in names])),
                               'field_s_median': float(np.median([tm[v][n]['field_s'] for n in names]))}
    res['skips'] = {v: {'timing_clips': sum(n in tm[v] for n in want), 'plans_complete_clips': res['probes'][v]['clips'],
                        'max_P': int(max(tm[v][n]['P'] for n in want if n in tm[v])), 'max_offset': int(max(tm[v][n]['offset'] for n in want if n in tm[v])),
                        'skipped_listed': [r for r in tm[v].values() if r.get('skipped') or r.get('skip')]} for v in ['V3', 'V4']}

    # Verdicts W1-W6, as written in the addendum.
    V = {}
    w1 = tau_verdict(res['shift']['V3-V1']); t10 = res['variants_own_curve']['V3'][E10]['tau_eff'][0]
    w1['parts']['V3_E10_tau_eff_in_0.80_0.90'] = .80 <= t10 <= .90
    w1['verdict'] = 'PASS' if all(w1['parts'].values()) else 'FAIL'; V['W1'] = w1
    a1, a3 = tau_verdict(res['shift']['V4-V1']), tau_verdict(res['shift']['V4-V3'])
    V['W2'] = {'vs_V1': a1, 'vs_V3': a3, 'verdict': 'PASS' if a1['verdict'] == a3['verdict'] == 'PASS' else 'FAIL'}
    pv = res['plans_vs_V1_E10']; parts = {'V3_lt_0.15m': pv['V3']['median_of_clip_medians_m'] < .15, 'V4_lt_0.20m': pv['V4']['median_of_clip_medians_m'] < .20}
    V['W3'] = {'parts': parts, 'verdict': 'PASS' if all(parts.values()) else 'FAIL'}
    rz = res['reasoning']; parts = {v: (rz[v]['fraction_identical'] is not None and rz[v]['fraction_identical'] >= .90) for v in ['V3', 'V4']}
    parts.update({f'{v}_P_upper_bound_ge_0.90': rz[v]['fraction_P_equal_400'] >= .90 for v in ['V3', 'V4']})
    V['W4'] = {'parts': parts, 'verdict': 'PASS' if all(parts.values()) else 'FAIL',
               'basis': f"exact text on {rz['V3']['clips_with_pytorch_text']} clips with a local PyTorch text; on 400 only the prefix length P (necessary condition)"}
    sr = res['spread_ratio_vs_V1']; parts = {f'{v}_{c}': .98 <= sr[v][c]['ratio_of_means'] <= 1.02 for v in ['V3', 'V4'] for c in [REFERENCE, E10]}
    V['W5'] = {'parts': parts, 'verdict': 'PASS' if all(parts.values()) else 'FAIL',
               'sensitivity_geomean_in_range': {f'{v}_{c}': .98 <= sr[v][c]['geomean_per_clip'] <= 1.02 for v in ['V3', 'V4'] for c in [REFERENCE, E10]}}
    sk = res['skips']; parts = {f'{v}_{k}': x for v in ['V3', 'V4'] for k, x in
                                [('all_400_plans', sk[v]['plans_complete_clips'] == len(want)), ('all_400_timed', sk[v]['timing_clips'] == len(want)),
                                 ('none_listed_skipped', not sk[v]['skipped_listed']), ('maxP_plus_256_le_3456', sk[v]['max_P'] + MAX_NEW <= L)]}
    V['W6'] = {'parts': parts, 'verdict': 'PASS' if all(parts.values()) else 'FAIL'}
    res['verdicts'] = V

    a.out.mkdir(parents=True, exist_ok=True); (a.out / 'summary.json').write_text(json.dumps(res, indent=2))
    (a.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (a.out / 'source_snapshot_thor_analysis.py').write_bytes(Path(ta.__file__).read_bytes())
    for v, e in res['variants_own_curve'].items():
        print(f'== {v} ' + '; '.join(f"{s.split('_')[0]} {x['tau_eff'][0]:.4f} [{x['tau_eff'][1]:.4f},{x['tau_eff'][2]:.4f}] ade1 {x['ade1']['residual'][0]:+.3f}{'≡' if x['ade1']['equivalent'] else ''} flip {x['flip']['residual'][0]:+.4f}{'≡' if x['flip']['equivalent'] else ''}" for s, x in e.items() if s in [E10, E4]))
    for k, sh in res['shift'].items():
        print(f'== {k}: ' + '; '.join(f"{s.split('_')[0]} {x['dtau_eff']:+.4f} ci95 [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] ci90 [{x['ci90'][0]:+.4f},{x['ci90'][1]:+.4f}]{'≡' if x['equivalent'] else ''}" for s, x in sh.items() if s in SAMPLERS))
    print('addendum2 reproduced', res.get('addendum2_reproduced')); print(json.dumps({k: res[k] for k in ['plans_vs_V1_E10', 'spread_ratio_vs_V1', 'reasoning', 'skips']}, indent=1))
    for k, v in V.items(): print(k, v['verdict'], v.get('parts'))


if __name__ == '__main__': main()
