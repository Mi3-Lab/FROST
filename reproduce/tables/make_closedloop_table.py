"""Closed-loop table (main text, label tab:closedloop): AlpaSim, Alpamayo-1.5, five configurations on the common scenes.

Registered entries are read from the stored analyses:
  - incident rates and scene-paired exact McNemar tests: results/closedloop_analysis_v1_20261028/summary.json
    (frost/analysis/closedloop_analysis.py; pairs C1, C2 unmatched/matched, C3 unmatched/matched)
  - paired RMS-jerk differences with 95% scene-bootstrap intervals: results/closedloop_comfort_v1_20261028/summary.json
    (frost/analysis/closedloop_comfort.py)
Layout (v7.1b): one row per comparison, no dashes. Exploratory entries (dagger), not in the registered pair list: the reference at tau=0.84 against tau=1 (incidents and jerk); the matched-temperature incident tests are stored (C2/C3 matched).
  - jerk: the comfort analysis's own estimator on its stored per-rollout values (per_rollout.json): the same common
    scenes, the same 2000 scene-bootstrap draws (np.random.default_rng(20261028).integers(0, n, (2000, n))), drawn once
    for all pairs as in closedloop_comfort.main. The script first reproduces every stored jerk test exactly.
  - incident McNemar: frost.analysis.closedloop_analysis.scene_table on the stored rollouts (artifacts/alpasim/closedloop_v1)
    with the same exact binomial test. This needs polars and the raw rollouts, so it is computed with --recompute and
    cached in results/closedloop_exploratory.json; the cache also re-derives every stored incident test as a check.

Run from the repo root:
  python reproduce/tables/make_closedloop_table.py --recompute   # refresh the cache (needs polars and $FROST_ARTIFACTS/alpasim/closedloop_v1)
  python reproduce/tables/make_closedloop_table.py               # write the table from results/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro_paths import RESULTS, TABLES as OUT  # noqa: E402
CL = RESULTS / 'closedloop_analysis_v1_20261028/summary.json'
CF = RESULTS / 'closedloop_comfort_v1_20261028'
from repro_paths import ARTIFACTS, REPO  # noqa: E402
GRID = ARTIFACTS / 'alpasim/closedloop_v1'
CACHE = RESULTS / 'closedloop_exploratory.json'
REF = 'heun32_t1.0'
BOOT, SEED = 2000, 20261028  # closedloop_comfort defaults


def num(x, d):
    t = f'{x:.{d}f}'
    if float(t) == 0: t = f'{0:.{d}f}'
    return t.replace('-', '$-$')


def jerk_tests():
    """Comfort estimator on per_rollout.json; asserts the stored jerk tests are reproduced, returns tau=0.84 vs 1."""
    pr = json.loads((CF / 'per_rollout.json').read_text());cf = json.loads((CF / 'summary.json').read_text())
    scenes = sorted(set.intersection(*[set(t) for t in pr.values()]));assert len(scenes) == cf['scenes']
    draws = np.random.default_rng(SEED).integers(0, len(scenes), (BOOT, len(scenes)))

    def test(a, b):
        d = np.array([pr[a][s]['jerk_rms'] - pr[b][s]['jerk_rms'] for s in scenes]);bt = d[draws].mean(1)
        return {'a': a, 'b': b, 'metric': 'jerk_rms', 'diff': float(d.mean()), 'diff_ci95': np.quantile(bt, [.025, .975]).tolist(),
                'mean_a': float(np.mean([pr[a][s]['jerk_rms'] for s in scenes])), 'mean_b': float(np.mean([pr[b][s]['jerk_rms'] for s in scenes]))}
    for t in cf['tests']:
        if t['metric'] == 'jerk_rms':
            r = test(t['a'], t['b']);assert abs(r['diff'] - t['diff']) < 1e-12 and np.allclose(r['diff_ci95'], t['diff_ci95'], atol=1e-12), t['label']
    return test('heun32_t0.84', REF)


def recompute_incidents():
    """Exploratory McNemar (tau=0.84 vs 1) with closedloop_analysis.scene_table; checks the stored incident tests."""
    sys.path.insert(0, str(REPO))
    from scipy.stats import binomtest
    from frost.analysis.closedloop_analysis import scene_table
    tabs = {}
    for run in sorted(GRID.glob('*_t*')):
        if '.failed' in run.name or not run.is_dir(): continue
        t = scene_table(run)
        if t: tabs[run.name] = t
    common = sorted(set.intersection(*(set(t) for t in tabs.values())))
    cl = json.loads(CL.read_text());assert len(common) == cl['common_scenes']

    def test(a, b, metric='incident'):
        oa = sum(tabs[a][s][metric] and not tabs[b][s][metric] for s in common);ob = sum(tabs[b][s][metric] and not tabs[a][s][metric] for s in common)
        return {'a': a, 'b': b, 'metric': metric, 'only_a': int(oa), 'only_b': int(ob),
                'rate_a': float(np.mean([tabs[a][s][metric] for s in common])), 'rate_b': float(np.mean([tabs[b][s][metric] for s in common])),
                'p_exact_mcnemar': float(binomtest(min(oa, ob), oa + ob).pvalue) if oa + ob else 1.}
    for t in cl['tests']:
        r = test(t['a'], t['b'], t['metric'])
        assert (r['only_a'], r['only_b']) == (t['only_a'], t['only_b']) and abs(r['p_exact_mcnemar'] - t['p_exact_mcnemar']) < 1e-12, t
    out = {'source': 'frost.analysis.closedloop_analysis.scene_table on artifacts/alpasim/closedloop_v1; exact binomial McNemar as in closedloop_analysis.main',
           'common_scenes': len(common), 'stored_tests_reproduced': len(cl['tests']),
           'exploratory': [test('heun32_t0.84', REF, m) for m in ('incident', 'collision_at_fault', 'offroad')]}
    CACHE.write_text(json.dumps(out, indent=1));print('wrote', CACHE)


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--recompute', action='store_true');args = ap.parse_args()
    if args.recompute: recompute_incidents()
    cl = json.loads(CL.read_text());cf = json.loads((CF / 'summary.json').read_text());ex = json.loads(CACHE.read_text())
    assert ex['common_scenes'] == cl['common_scenes'] == cf['scenes']
    inc = lambda a, b: next(t for t in cl['tests'] if t['a'] == a and t['b'] == b and t['metric'] == 'incident')
    jerk = lambda a, b: next(t for t in cf['tests'] if t['a'] == a and t['b'] == b and t['metric'] == 'jerk_rms')
    j84 = jerk_tests();i84 = next(t for t in ex['exploratory'] if t['metric'] == 'incident')
    assert abs(i84['rate_a'] - cl['rates']['heun32_t0.84']['incident']) < 1e-12 and abs(i84['rate_b'] - cl['rates'][REF]['incident']) < 1e-12
    D = '$^\\dagger$'
    jcell = lambda t: f"{num(t['diff'], 3)}\\,{{\\scriptsize[{num(t['diff_ci95'][0], 3)},\\,{num(t['diff_ci95'][1], 3)}]}}"
    ex_inc = {(t['a'], t['b']): t for t in ex['exploratory'] if t['metric'] == 'incident'}

    def inc_test(a, b):
        """Stored registered McNemar test if it exists, else the exploratory one computed with the same function."""
        t = next((t for t in cl['tests'] if t['a'] == a and t['b'] == b and t['metric'] == 'incident'), None)
        return (t, False) if t is not None else (ex_inc[a, b], True)

    def jerk_test(a, b):
        t = next((t for t in cf['tests'] if t['a'] == a and t['b'] == b and t['metric'] == 'jerk_rms'), None)
        if t is not None: return t, False
        assert (a, b) == ('heun32_t0.84', REF);return j84, True
    name = {REF: '$\\tau{=}1$', 'heun32_t0.84': '$\\tau{=}0.84$', 'heun32_t0.65': '$\\tau{=}0.65$', 'euler10_t1.0': '$E_{10}$', 'euler4_t1.0': '$E_4$'}
    groups = [('Cooling the reference', [('heun32_t0.84', REF), ('heun32_t0.65', REF)]),
              ('Samplers vs.\\ unit temperature', [('euler10_t1.0', REF), ('euler4_t1.0', REF)]),
              ('Samplers vs.\\ matched temperature', [('euler10_t1.0', 'heun32_t0.84'), ('euler4_t1.0', 'heun32_t0.65')])]
    rate = lambda c: f"{100 * cl['rates'][c]['incident']:.0f}"
    lines, printed = [], []
    for g, (head, pairs) in enumerate(groups):
        if g: lines.append('\\midrule')
        lines.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\')
        for a, b in pairs:
            it, ie = inc_test(a, b);jt, je = jerk_test(a, b)
            assert abs(it['rate_a'] - cl['rates'][a]['incident']) < 1e-12 and abs(it['rate_b'] - cl['rates'][b]['incident']) < 1e-12
            dag = D if (ie or je) else ''
            cells = [f'\\hspace{{0.3em}}{name[a]} vs.\\ {name[b]}{dag}', f'{rate(a)}/{rate(b)}',
                     f"{it['p_exact_mcnemar']:.2f} ({it['only_a']}/{it['only_b']})", jcell(jt)]
            lines.append(' & '.join(cells) + ' \\\\');printed.append((a, b, ie, je, cells))
    body = '\n    '.join(lines)
    n = cl['common_scenes']
    (OUT / 'tab_closedloop.tex').write_text(rf"""% Generated by paper/tables/make_closedloop_table.py; do not edit by hand.
\begin{{table}}[t]
  \caption{{\textbf{{In closed loop, temperature changes comfort; incident rates do not separate.}} Alpamayo-1.5, {n} AlpaSim scenes, one rollout per scene and configuration; each row compares configuration $a$ with $b$. Inc.: rate (\%) of at-fault collision or off-road events in $a$/$b$. McNemar: exact scene-paired test, with discordant scenes (incident only in $a$ / only in $b$). $\Delta$ jerk: paired difference in RMS jerk, $a-b$ (m/s$^3$), with 95\% scene-bootstrap interval. $^\dagger$: exploratory comparison outside the registered pairs, same tests and rollouts.}}
  \label{{tab:closedloop}}
  \centering\footnotesize\setlength{{\tabcolsep}}{{2pt}}
  \begin{{tabular}}{{@{{}}lccc@{{}}}}
    \toprule
    $a$ vs.\ $b$ & Inc.\,\% & McNemar $p$ & $\Delta$ jerk \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")
    mac = [f'% Generated by paper/tables/make_closedloop_table.py; exploratory (tau=0.84 vs tau=1).',
           f"\\newcommand{{\\CLJerkTauEightFour}}{{{jcell(j84)}}}",
           f"\\newcommand{{\\CLIncPTauEightFour}}{{{i84['p_exact_mcnemar']:.2f}}}",
           f"\\newcommand{{\\CLIncPETen}}{{{inc('euler10_t1.0', REF)['p_exact_mcnemar']:.2f}}}",
           f"\\newcommand{{\\CLIncPEFour}}{{{inc('euler4_t1.0', REF)['p_exact_mcnemar']:.2f}}}",
           f"\\newcommand{{\\CLIncPTauSixFive}}{{{inc('heun32_t0.65', REF)['p_exact_mcnemar']:.2f}}}"]
    (OUT / 'numbers_closedloop.tex').write_text('\n'.join(mac) + '\n')
    print((OUT / 'tab_closedloop.tex').read_text());print('\n'.join(mac))
    print(f"jerk 0.84-1: mean {j84['mean_a']:.4f} vs {j84['mean_b']:.4f}; incidents 0.84 vs 1: {i84['only_a']}/{i84['only_b']} p={i84['p_exact_mcnemar']:.3f}")


if __name__ == '__main__':
    main()
