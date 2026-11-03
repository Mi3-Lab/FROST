"""Complete FROST evidence atlas, intended for the supplement.

All 27 model-sampler pairs retain their exact JSON values, confidence
intervals, sample counts, margins, verdicts, upper-grid edges and censoring.
The plotting columns are model/sampler, effective temperature, and residuals
of ADE1, decision flips and PDMS divided by their original equivalence margins.
Symbols retain each marginal 90% equivalence verdict. Drawn residual intervals
are 95% for Alpamayo/Heun and 90% for other families, as in the original atlas.
A plotted 95% interval may therefore cross a margin for an equivalent symbol.
No endpoint is clipped or winsorized. All out-of-range cases stay visible.

Only read-only make_tables.load_pooled() is imported; table writers never run.
The optional --a19 keeps the original converged-reference sensitivity behavior.
Run on CPU: MPLBACKEND=Agg python reproduce/figures/make_fig_frost_atlas.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tables'))
import make_tables as mt  # noqa: E402  (load_pooled: the exact Alpamayo + Heun-control entries behind tab:matched)

ART = fs.ART
NAV = {f: ART / f'navsim_analysis_recogdrive_{f}_addendum2_v1_20261028' / 'summary.json' for f in ('il', 'rl')}
DD = ART / 'navsim_analysis_diffusiondrive_v3_20261029' / 'summary.json'
QD = ART / 'qwendrive_analysis_v2_20261027' / 'summary.json'  # margins only (FlashDrive rows)
QD3 = ART / 'qwendrive_v3_analysis_20261102' / 'summary.json'
QDH = ART / 'qwendrive_v3_heun_analysis_20261102' / 'summary.json'  # registered Heun control (addendum 1)  # Qwen-Drive rows: registered v3, 400 held-out clips
QD6 = ART / 'qwendrive_v3_e6_analysis_20261102' / 'summary.json'  # registered E6 (addendum 2)
FD = ART / 'flashdrive_analysis_nvidia_v2_20261026' / 'summary.json'
FDF = ART / 'flashdrive_flips_v1_20261028' / 'summary.json'

TEMP, OWN, NONE = 'temperature', 'own', 'none'
SECTION = {TEMP: 'Temperature-like: residuals inside every margin at $\\tau_{\\mathrm{eff}}$',
           OWN: 'Not equivalent: a residual interval reaches past a margin',
           NONE: 'No matching temperature: $\\tau_{\\mathrm{eff}}$ outside the reference grid'}
COLOR = {TEMP: fs.BLUE, OWN: fs.VERMILLION, NONE: fs.GRAY}
MARK = {TEMP: 'o', OWN: 'D', NONE: 's'}
MODEL = {'alpamayo15': 'Alpamayo-1.5', 'alpamayo_r1': 'Alpamayo-R1', 'alpamayo2': 'Alpamayo-2'}

# Rows (model, sampler, display). Edit here to add or drop rows; the section each row lands in comes from the JSON.
ROWS = [('alpamayo15', 'euler10_t1.0', '$E_{10}$ (released)'), ('alpamayo15', 'euler6_t1.0', '$E_6$'), ('alpamayo15', 'euler4_t1.0', '$E_4$'),
        ('alpamayo_r1', 'euler10_t1.0', '$E_{10}$ (released)'), ('alpamayo_r1', 'euler6_t1.0', '$E_6$'), ('alpamayo_r1', 'euler4_t1.0', '$E_4$'),
        ('alpamayo2', 'euler10_t1.0', '$E_{10}$ (released)'), ('alpamayo2', 'euler6_t1.0', '$E_6$'), ('alpamayo2', 'euler4_t1.0', '$E_4$'),
        ('flashdrive', 'flashcache8_t1.0', 'cache, 8 steps'),
        ('recog_il', 'ddim4_lead_eta0_t1.0', 'DDIM $E_4$'), ('recog_il', 'ddim10_lead_eta0_t1.0', 'DDIM $E_{10}$'),
        ('alpamayo15', 'heun5_t1.0', 'Heun-5'), ('alpamayo_r1', 'heun5_t1.0', 'Heun-5'), ('alpamayo2', 'heun5_t1.0', 'Heun-5'),
        ('recog_il', 'ddim2_lead_eta0_t1.0', 'DDIM $E_2$'), ('recog_il', 'release5_lead_eta1_t1.0', 'release ($\\eta{=}1$)'),
        ('recog_rl', 'ddim4_lead_eta0_t1.0', 'DDIM $E_4$'), ('recog_rl', 'ddim5_lead_eta0_t1.0', 'DDIM $E_5$'),
        ('recog_rl', 'release5_lead_eta1_t1.0', 'release ($\\eta{=}1$)'),
        ('qwen_rl', 'euler10_t1.0', '$E_{10}$'), ('qwen_rl', 'euler6_t1.0', '$E_6$'), ('qwen_rl', 'euler4_t1.0', '$E_4$'), ('qwen_rl', 'heun5_t1.0', 'Heun-5'),
        ('recog_rl', 'ddim10_lead_eta0_t1.0', 'DDIM $E_{10}$'), ('recog_rl', 'ddim20_lead_eta0_t1.0', 'DDIM $E_{20}$'),
        ('diffusiondrive', 'release2_lead_eta0_t1.0', 'release (2 steps)')]
# What the paper states for each row (scorecard tab:scorecard, Sec. 4, App. 'A second family'); used only as a check.
PAPER = {('alpamayo15', 'euler6_t1.0'): TEMP, ('alpamayo_r1', 'euler6_t1.0'): OWN, ('alpamayo2', 'euler6_t1.0'): TEMP, ('qwen_rl', 'euler6_t1.0'): TEMP,  # Table 1 (90% rule); R1 E6 flip residual +1.3 margins
         ('alpamayo15', 'euler4_t1.0'): TEMP, ('alpamayo_r1', 'euler4_t1.0'): OWN, ('alpamayo2', 'euler4_t1.0'): OWN,  # scorecard: E4 flips 1/3
         ('alpamayo15', 'euler10_t1.0'): TEMP, ('alpamayo_r1', 'euler10_t1.0'): TEMP, ('alpamayo2', 'euler10_t1.0'): TEMP,
         ('flashdrive', 'flashcache8_t1.0'): OWN, ('recog_il', 'ddim4_lead_eta0_t1.0'): TEMP, ('recog_il', 'ddim10_lead_eta0_t1.0'): TEMP,
         ('alpamayo15', 'heun5_t1.0'): OWN, ('alpamayo_r1', 'heun5_t1.0'): OWN, ('alpamayo2', 'heun5_t1.0'): OWN,
         ('recog_il', 'ddim2_lead_eta0_t1.0'): OWN, ('recog_il', 'release5_lead_eta1_t1.0'): OWN,
         ('recog_rl', 'ddim4_lead_eta0_t1.0'): OWN, ('recog_rl', 'ddim5_lead_eta0_t1.0'): OWN, ('recog_rl', 'release5_lead_eta1_t1.0'): OWN,
         ('qwen_rl', 'euler10_t1.0'): TEMP, ('qwen_rl', 'euler4_t1.0'): OWN, ('qwen_rl', 'heun5_t1.0'): NONE, ('recog_rl', 'ddim10_lead_eta0_t1.0'): NONE, ('recog_rl', 'ddim20_lead_eta0_t1.0'): NONE,
         ('diffusiondrive', 'release2_lead_eta0_t1.0'): NONE}
METRICS = [('ade1', 'ADE$_1$'), ('flip', 'flips'), ('pdms', 'PDMS')]
FIN = lambda x: x is not None and not (isinstance(x, float) and math.isnan(x))


def row_alpamayo(stats, model, sampler, a19=None):
    e = stats['models'][model][sampler];src = 'pooled'
    if a19 is not None and model == 'alpamayo15' and sampler == 'euler10_t1.0':
        e = a19['matched']['heun64']['euler10_t1.0'];src = 'a19'
    t = e['tau_eff']
    return dict(group=MODEL[model], n=f"{stats['clips']} clips", tau=t[0], lo=t[1], hi=t[2], ci=95, in_range=e['in_range'],
                edge=not e['in_range'] and FIN(t[0]), eq={m: e[m]['equivalent'] for m in ('ade1', 'flip')},
                res={m: e[m]['residual'] for m in ('ade1', 'flip')}, margins=stats['margins'], src=src)


def row_navsim(d, sampler, group):
    e = d['samplers'][sampler];ok = e['in_range'] and FIN(e['tau_eff'])
    r = dict(group=group, n=f"{d['scenes']} scenes", ci=95, in_range=ok, edge=False, margins=d['margins'])
    if ok:
        r.update(tau=e['tau_eff'], lo=e['tau_eff_ci95'][0], hi=e['tau_eff_ci95'][1],
                 eq={m: e[m]['equivalent'] for m in ('ade1', 'flip', 'pdms')},
                 res={m: [e[m]['residual'], *e[m]['residual_ci90']] for m in ('ade1', 'flip', 'pdms')})
    else:  # make_tables.navsim_table's rule for the reason
        e50 = d['samplers'].get('ddim50_lead_eta0_t1.0')
        r.update(tau=None, why=None if e50 is None else ('above' if e['spread'] >= e50['spread'] else 'below'))
        c = e.get('tau_eff_ci95_censored')
        if c is not None:  # DiffusionDrive: out-of-range resamples kept as tau_eff > 1 (audit A3)
            assert not FIN(c[1]);r.update(lo=c[0], hi=None, boot_in_range=e['boot_in_range'], why='censored')
    return r


def load(a19_path):
    stats = mt.load_pooled();a19 = json.loads(Path(a19_path).read_text()) if a19_path else None
    nav = {f: json.loads(p.read_text()) for f, p in NAV.items()};dd = json.loads(DD.read_text())
    qd = json.loads(QD.read_text());fd = json.loads(FD.read_text());ff = json.loads(FDF.read_text())
    rows = []
    for model, sampler, disp in ROWS:
        if model in MODEL:
            r = row_alpamayo(stats, model, sampler, a19)
        elif model in ('recog_il', 'recog_rl'):
            fam = model[-2:];r = row_navsim(nav[fam], sampler, f'ReCogDrive-{fam.upper()}')
        elif model == 'diffusiondrive':
            r = row_navsim(dd, sampler, 'DiffusionDrive')
        elif model == 'qwen_rl':
            q3 = json.loads((QDH if sampler.startswith('heun') else QD6 if sampler == 'euler6_t1.0' else QD3).read_text());e = q3['horizons']['native_5s']['samplers'][sampler]
            r = dict(group='Qwen-Drive (RL)', n=f"{q3['clips']} clips", tau=None if e.get('above_grid') else e['tau_eff'], why='above' if e.get('above_grid') else None, lo=e['tau_eff_ci90'][0], hi=e['tau_eff_ci90'][1],
                     ci=90, in_range=e['in_range'], edge=False, eq={m: e[m]['equivalent'] for m in ('ade1', 'flip')},
                     res={m: [e[m]['residual'], *e[m]['residual_ci90']] for m in ('ade1', 'flip')}, margins=q3['margins'])
        elif model == 'flashdrive':
            c = fd['samplers'][sampler];f = ff['flashdrive_probe_nvidia_v1']['samplers'][sampler];M = qd['margins']
            assert abs(c['tau_eff'] - f['tau_eff']) < 1e-4  # same matched temperature in both FlashDrive analyses
            # FlashDrive artifacts store residual intervals but no TOST flag; equivalence = 90% interval inside the margin,
            # the rule paper_numbers applies (and the 0.05 m / 2 pt margins the FlashDrive paragraphs quote).
            inside = lambda ci, m: -m < ci[0] and ci[1] < m
            r = dict(group='FlashDrive (Alp.-1.5)', n=f"{fd['clips']} clips", tau=c['tau_eff'], lo=c['tau_ci90'][0], hi=c['tau_ci90'][1],
                     ci=90, in_range=c['in_range'], edge=False, margins=M,
                     eq={'ade1': inside(c['residual_ci90'], M['ade1']), 'flip': inside(f['flip_resid_ci90'], M['flip'])},
                     res={'ade1': [c['residual'], *c['residual_ci90']], 'flip': [f['flip_resid'], *f['flip_resid_ci90']]},
                     e4=fd['samplers']['euler4_t1.0'])
        r.update(key=(model, sampler), disp=disp)
        r['json'] = NONE if r['tau'] is None or not all(FIN(v[0]) for v in r['res'].values()) else (TEMP if all(r['eq'].values()) else OWN)
        rows.append(r)
    return rows


def reason(r):
    """Short reason. For a failed TOST: 'outside the margin' when the whole stored residual interval lies beyond it
    (95% for the pooled Alpamayo/Heun entries, 90% elsewhere), 'interval crosses the margin' otherwise."""
    if r['json'] == NONE:
        if r['why'] == 'censored':
            return f"$\\tau_{{\\mathrm{{eff}}}}$ interval runs past 1; {100 * r['boot_in_range']:.0f}% of resamples in range"
        return 'spreads more than the reference at $\\tau{=}1$' if r['why'] == 'above' else 'spreads less than the coldest reference'
    if r['json'] == TEMP:
        return 'inside the PDMS, error and flip margins' if 'pdms' in r['eq'] else 'inside the error and flip margins'
    M = r['margins'];outs, crosses, bits = [], [], []
    for m, noun, adj in (('flip', 'flips', 'flip'), ('ade1', 'error', 'error'), ('pdms', 'PDMS', 'PDMS')):
        if m in r['eq'] and not r['eq'][m]:
            v = r['res'][m];(outs.append(noun) if v[1] > M[m] or v[2] < -M[m] else crosses.append(adj))
    if r['key'][0] == 'flashdrive':
        e4 = r['e4'];bits.append('$\\tau_{\\mathrm{eff}}$ of $E_4$' if e4['tau_ci90'][0] <= r['hi'] and r['lo'] <= e4['tau_ci90'][1] else '')
        if crosses == ['error']:  # the dagger note: how far past the margin, and E4 on the same clips
            crosses = [];bits.append(f"error interval ends {r['res']['ade1'][2] - M['ade1']:.3f} m past the margin")
    if outs:
        bits.append(f"{' and '.join(outs)} outside the margin")
    if crosses:
        bits.append(f"{' and '.join(crosses)} interval{'s cross' if len(crosses) > 1 else ' crosses'} the margin")
    if r['res']['ade1'][2] < 0:
        bits.append('lower error')
    return '; '.join(b for b in bits if b)


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--a19', default=None, help='Addendum 19 results.json (E10 vs Heun-64)')
    args = ap.parse_args()
    rows = load(args.a19)
    for r in rows:  # check against the paper's stated verdicts
        r['verdict'] = r['json'];r['dagger'] = r['key'][0] == 'flashdrive'  # caption note: E4 on the same clips
        if r['key'][1] == 'euler6_t1.0':
            assert r['json'] == PAPER[r['key']], (r['key'], r['json'])  # registered E6 verdicts are hard checks
        if r['json'] != PAPER[r['key']]:
            print(f"WARNING verdict mismatch {r['key']}: JSON rule -> {r['json']}, paper -> {PAPER[r['key']]} (drawn with the JSON verdict)")
        if r['dagger']:
            e4 = r['e4'];print(f"NOTE FlashDrive dagger: cache ADE1 resid hi90 {r['res']['ade1'][2]:.4f}; E4 same clips hi90 {e4['residual_ci90'][1]:.4f} (past margin: {e4['residual_ci90'][1] > r['margins']['ade1']})")
    order = [TEMP, OWN, NONE]
    rows = sorted(rows, key=lambda r: order.index(r['verdict']))
    y, lines, ypos = 0., [], []
    for verdict in order:
        lines.append((y, verdict));y += 1.15
        for r in rows:
            if r['verdict'] == verdict:
                ypos.append((y, r));y += 1
        y += 0.55
    H = y - 0.55
    fs.style()
    fig = plt.figure(figsize=(fs.TEXT, H * 0.16 + 0.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.2, 1.35, 1, 1, 1],
                          wspace=0.25, left=0, right=1, top=1, bottom=0)
    axL, axT, *resaxes = [fig.add_subplot(gs[i]) for i in range(5)]
    axes = [axL, axT, *resaxes]
    for ax in axes:
        ax.set_ylim(H - 0.45, -0.8)
        ax.tick_params(labelsize=7)
    axL.set_xlim(0, 1);axL.axis('off')
    axT.set_xlim(0.28, 1.12);axT.set_xticks([0.4, 0.7, 1.0], ['0.4', '0.7', '1.0'])
    axT.set_xlabel(r'$\tau_{\mathrm{eff}}$', fontsize=8)
    axT.text(1.06, -0.84, '>1', ha='center', va='bottom', fontsize=7)
    axT.axvspan(1, 1.12, color=fs.BAND, alpha=0.22, lw=0)
    for ax in axes[1:]:
        ax.spines['left'].set_visible(False);ax.tick_params(left=False, labelleft=False)
    # Residual/margin keeps all original endpoints and permits a common equivalence region.
    for ax, (metric, title) in zip(resaxes, METRICS):
        vals = [v / r['margins'][metric] for r in rows if r['tau'] is not None
                and metric in r.get('res', {}) for v in r['res'][metric]]
        low, high = min([-1.4, *vals]), max([1.4, *vals])
        pad = (high - low) * 0.07
        ax.set_xlim(low - pad, high + pad)
        ax.axvline(0, color=fs.GRAY, lw=0.4)
        for boundary in [-1, 1]:
            ax.axvline(boundary, color=fs.GRAY, ls='--', lw=0.55)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('residual / margin', fontsize=7)
        ax.set_xticks([-1, 0, 1] if high < 3 else [-1, 1, round(high)])
    heads = {TEMP: 'Equivalent at matched temperature', OWN: 'Equivalence not established', NONE: 'No matching temperature in the grid'}
    for yh, verdict in lines:
        axL.text(0, yh, heads[verdict], fontsize=8, weight='bold', va='center', color=COLOR[verdict], clip_on=False)
        for ax in axes[1:]:
            ax.axhline(yh + 0.53, lw=0.5, color=COLOR[verdict], alpha=0.6)
    for yy, r in ypos:
        col = COLOR[r['verdict']]
        group = r['group'].replace('FlashDrive (Alp.-1.5)', 'FlashDrive').replace('Qwen-Drive (RL)', 'Qwen-Drive RL')
        sampler = r['disp'].replace(' (released)', '').replace('DDIM ', '').replace('cache, 8 steps', 'cache')
        axL.text(0, yy, group, fontsize=7.5, va='center')
        axL.text(1, yy, sampler + (r' $\dagger$' if r['dagger'] else ''), fontsize=7.5, va='center', ha='right')
        if r['tau'] is not None:
            axT.plot([r['lo'], r['hi']], [yy, yy], color=col, lw=1)
            axT.plot(r['tau'], yy, MARK[r['verdict']], ms=3.8, color=col, mec='white', mew=0.3)
            if r['edge']:
                axT.annotate('edge', (r['tau'], yy), xytext=(3, 0), textcoords='offset points', fontsize=6.3, va='center')
        elif r['why'] == 'censored':
            axT.annotate('', xy=(1.12, yy), xytext=(r['lo'], yy), arrowprops=dict(arrowstyle='-|>', color=col, lw=1, shrinkA=0, shrinkB=0, mutation_scale=7))
            axT.plot(r['lo'], yy, '|', ms=4, color=col)
            axT.annotate(f"{100*r['boot_in_range']:.0f}% in range", (0.73, yy), fontsize=6.3, va='center', ha='right')
        else:
            axT.plot(1.09 if r['why'] == 'above' else 0.29, yy,
                     '>' if r['why'] == 'above' else '<', ms=4, color=col)
        for ax, (metric, _) in zip(resaxes, METRICS):
            if r['tau'] is None or metric not in r.get('res', {}):
                ax.text(0, yy, '—', ha='center', va='center', color=fs.GRAY, fontsize=7)
                continue
            v, lo, hi = [value / r['margins'][metric] for value in r['res'][metric]]
            ok = r['eq'][metric];c = fs.BLUE if ok else fs.VERMILLION
            ax.plot([lo, hi], [yy, yy], color=c, lw=1)
            ax.plot(v, yy, 'o' if ok else 'X', ms=3.7 if ok else 4.2, color=c, mec='white', mew=0.3)
    handles = [Line2D([], [], marker='o', ls='', color=fs.BLUE, label='equivalent (90% test)', ms=4),
               Line2D([], [], marker='X', ls='', color=fs.VERMILLION, label='equivalence not established', ms=4),
               Line2D([], [], color=fs.GRAY, lw=1, label='stored residual interval')]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.085), fontsize=7,
               handlelength=1.3, columnspacing=1.4)
    fs.save(fig, 'fig_frost_atlas.pdf')
    for yy, r in ypos:
        print(f"{r['verdict']:11s} {r['group']:28s} {r['key'][1]:26s} tau={r['tau']} [{r.get('lo')}, {r.get('hi')}] ({r['ci']}%) eq={r.get('eq')} src={r.get('src', '')}")


if __name__ == '__main__':
    main()
