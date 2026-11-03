"""Figure 4: cooling curves and sampler effects for four driving policies.

Two aligned rows show error and decision changes on the native plan horizon.
Alpamayo panels share limits; Qwen uses its labeled native scale in the same
regular grid. All curves, paired clip-bootstrap bands, effective-temperature
intervals and out-of-grid observations come from the existing analyses.
Qwen E6 is included only after its registered R2 reproducibility check passes.
Existing metrics and bootstrap definitions are reused; no model inference
or table writer is run by this plotting script.

Run on CPU: MPLBACKEND=Agg python reproduce/figures/make_fig_curves.py
    --rows ade1,flip --out fig_curves_main.pdf
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

MODELS = ['alpamayo15', 'alpamayo_r1', 'alpamayo2']
QV3 = 'qwendrive_v3_analysis_20261102'
QHEUN = 'qwendrive_v3_heun_analysis_20261102'  # Qwen-Drive Heun-3/5 control (protocols/qwen_drive.md, addendum 1)
QE6 = 'qwendrive_v3_e6_analysis_20261102'  # Qwen-Drive Euler-6 (protocols/qwen_drive.md, addendum 2); drawn once its summary exists and R2 passed
QBAND = 'qwendrive_v3_curve_band_20261102'  # per-tau paired clip bootstrap of the Qwen-Drive curve (frost.analysis.qwendrive_v3_curve_band)
MARGIN = {'ade1': 0.05, 'flip': 0.02}  # registered equivalence margins (metric units, before scale)
ROWS = {'ade1': (r'$\Delta$ADE$_1$ vs. $A_1$ (m)', 1.), 'flip': (r'decisions changed (%)', 100.),
        'minade6': (r'$\Delta$minADE$_6$ vs. $A_1$ (m)', 1.)}
BELOW_LABEL = {('alpamayo15', 'ade1', 1): '$E_{1,2}$', ('alpamayo_r1', 'ade1', 1): '$E_{1,2}$', ('alpamayo_r1', 'ade1', 2): None,
               ('alpamayo2', 'ade1', 1): '$E_{1,2}$', ('alpamayo2', 'ade1', 2): None}  # coincident in-panel markers share one label  # coincident markers share one label
MODEL_OFFSET = {('alpamayo15', 2): None, ('alpamayo_r1', 'heun5_t1.0'): (-4, 16)}  # in-range Euler label offsets that differ from LABEL_OFFSET
LABEL_OFFSET = {2: (-3, 4), 4: (-3, 4), 6: (-3, 4), 10: (-3, 4)}  # above-left: clear of the upper margin line (curves rise)


HEUN = {'alpamayo_r1': 'heun_controls_analysis_r1_v1_20261028', 'alpamayo2': 'heun_controls_analysis_a2_v1_20261028'}


def draw_margin(ax, tau, y, metric, scale):
    """Dashed gray lines at curve +- registered margin (as the WOD figure)."""
    if metric not in MARGIN:
        return
    d = MARGIN[metric] * scale
    for sgn in (1, -1):
        ax.plot(tau, np.asarray(y) + sgn * d, color='#A6ABB0', lw=0.55, ls=(0, (3, 2.4)), zorder=1)


def draw_below(ax, x, v, label, dy=0):
    """Sampler whose spread is below the grid: small left-pointing marker at the coldest grid point x, at the sampler's
    own metric value; no residual tick (no matched temperature). Deferred to finish_panel, which either draws it inside
    the panel or, when it would compress the panel, clips it at the edge with an arrow and its value."""
    ax._below = getattr(ax, '_below', []) + [(x, v, label, dy)]


CLIP_FRAC = 0.25  # below-grid points farther than this fraction of the panel's content range beyond its edge are clipped


def finish_row(axs, metric, fixed=None):
    """Common y limits for the panels of one row (autoscaled union, or `fixed`), then the deferred below-grid markers:
    inside when within CLIP_FRAC of the range beyond the edge (autoscaled rows only), else an edge arrow with the value."""
    if fixed is None:
        lims = []
        for ax in axs:
            ax.relim();ax.autoscale_view(scalex=False);lims.append(ax.get_ylim())
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        if metric == 'flip':
            lo = 0.
    else:
        lo, hi = fixed
    r = hi - lo;place = []
    for ax in axs:
        for x, v, label, dy in getattr(ax, '_below', []):
            ok = (lo - CLIP_FRAC * r <= v <= hi + CLIP_FRAC * r) if fixed is None else (lo <= v <= hi)
            place.append((ax, x, v, label, dy, ok))
    for ax, x, v, label, dy, ok in place:
        if ok and fixed is None:
            lo, hi = min(lo, v - 0.04 * r), max(hi, v + 0.04 * r)
    if fixed is None and any((not ok) and v > hi for _, _, v, _, _, ok in place):  # room under an edge arrow
        hi = max([hi] + [v + 0.14 * r for _, _, v, _, _, ok in place if ok])
    for ax in axs:
        ax.set_ylim(lo, hi)
    clipped = {}
    for ax, x, v, label, dy, ok in place:
        if ok:
            ax.plot(x, v, marker='<', ms=3.2, color=fs.BLUE, mec='white', mew=0.3, lw=0, zorder=4, clip_on=False)
            if label:
                ax.annotate(label, (x, v), xytext=(-3, dy), textcoords='offset points', ha='right', va='center', fontsize=6.5, color='#333333')
        else:
            clipped.setdefault((ax, x, v > hi), []).append((v, label))
    for (ax, x, up), pts in clipped.items():  # off-scale: arrow at the panel edge, value(s) next to it
        y = hi if up else lo
        ax.plot(x, y, marker='^' if up else 'v', ms=3.4, color=fs.BLUE, mec='white', mew=0.3, lw=0, zorder=5, clip_on=False)
        fmt = (lambda v: f'{v:+.2f}') if metric != 'flip' else (lambda v: f'{v:.0f}' if v >= 10 else f'{v:.1f}')
        text = ', '.join(f'{lab} {fmt(v)}' for v, lab in sorted(pts, reverse=up) if lab)
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=6.5, color='#333333')


def finish_panel(ax, metric, fixed=None):
    finish_row([ax], metric, fixed)


QFIXED = {'ade1': (-0.06, 0.06), 'flip': (0., 4.)}  # Qwen-Drive own scale: fills the panel, shows the margins


def draw_above(ax, v, m, label, label_off):
    """Sampler whose tau_eff is above the grid: right-pointing marker at tau = 1, residual tick from the curve there."""
    ax.plot([1., 1.], [m, v], color='#9CA3AA', lw=0.55, ls=(0, (1.5, 1.7)), zorder=2, solid_capstyle='butt')
    ax.plot(1., v, marker='>', ms=3.6, color=fs.VERMILLION, mec='white', mew=0.4, lw=0, zorder=4, clip_on=False)
    if label:
        ax.annotate(label, (1., v), xytext=label_off, textcoords='offset points', ha='right', va='center', fontsize=8)


def heun_entry(model, config, num):
    """Heun-k control of Alpamayo-R1 / -2 from the heun_controls analysis, in the layout of paper_numbers entries."""
    import json
    s = json.loads((fs.ART / HEUN[model] / 'summary.json').read_text())['samplers'][config]
    ref = num[model]['euler10_t1.0']
    return {'in_range': s['in_range'], 'tau_eff': [s['tau_eff'], *s['tau_eff_ci95']],
            'ade1': {'reference': ref['ade1']['reference'], 'sampler': s['ade1']['sampler'], 'matched': s['ade1']['matched']},
            'flip': {'reference': 0., 'sampler': s['flip']['sampler'], 'matched': s['flip']['matched']}}


def draw(ax, table, names, samples, num, model, metric, scale, label_euler, label_heun):
    tau, y, lo, hi = fs.curve(table, names, metric, samples, scale, paired=True)
    ax.axhline(0, color=fs.GRAY, lw=0.4, ls=':', zorder=0)
    ax.fill_between(tau, lo, hi, color=fs.BAND, alpha=0.6, lw=0, zorder=1)
    ax.plot(tau, y, color=fs.BLACK, lw=1.0, zorder=2)
    draw_margin(ax, tau, y, metric, scale)
    samplers = [(f'euler{k}_t1.0', f'$E_{{{k}}}$', fs.BLUE, 'o', k) for k in (1, 2, 4, 6, 10)]
    samplers += [(f'heun{k}_t1.0', f'Heun-{k}', fs.VERMILLION, 'D', None) for k in (3, 5)]
    for config, name, color, marker, k in samplers:
        e = num.get(config) if model == 'alpamayo15' or k is not None else (None if metric == 'minade6' else heun_entry(model, config, ALLNUM))
        if e is None:
            continue
        if not e['in_range']:
            if k is not None and e['tau_eff'][0] <= tau.min() + 1e-9:  # Euler below the grid: left edge (truncated tau_eff)
                v = (e[metric]['sampler'] - e[metric]['reference']) * scale
                draw_below(ax, tau.min(), v, BELOW_LABEL.get((model, metric, k), name), dy=-2 if (model, metric, k) == ('alpamayo_r1', 'ade1', 1) else 0)
                print(f"{model:11s} {config} below the grid; drawn at tau={tau.min():.1f}, {metric} d{v:+.3f} (no residual)")
            elif k is None and e['tau_eff'][0] >= 1.:  # Heun above the grid: drawn at tau = 1 (residual = effect)
                v = (e[metric]['sampler'] - e[metric]['reference']) * scale;m = (e[metric]['matched'] - e[metric]['reference']) * scale
                draw_above(ax, v, m, name if label_heun else None, (-5, 9) if config == 'heun5_t1.0' else (-5, 0))
            continue
        ref = e[metric]['reference'];x = e['tau_eff']
        v = (e[metric]['sampler'] - ref) * scale;m = (e[metric]['matched'] - ref) * scale
        ax.plot([x[0], x[0]], [m, v], color='#9CA3AA', lw=0.55, ls=(0, (1.5, 1.7)), zorder=2, solid_capstyle='butt')
        ax.errorbar(x[0], v, xerr=[[x[0] - x[1]], [x[2] - x[0]]], fmt=marker, ms=4.0 if marker == 'o' else 3.3,
                    color=color, mec='white', mew=0.4, elinewidth=0.65, capsize=0, zorder=4)
        if (k is not None and label_euler) or (k is None and label_heun):
            off = MODEL_OFFSET.get((model, config), MODEL_OFFSET.get((model, k), LABEL_OFFSET.get(k, (-6, 0) if config == 'heun3_t1.0' else (-4, 3))))
            if off is None:  # named by the adjacent below-grid label (Alpamayo-1.5: E_{1,2} at the left edge)
                continue
            ha = 'center' if off[0] == 0 else ('left' if off[0] > 0 else 'right')
            ax.annotate(name, (x[0], v), xytext=off, textcoords='offset points', ha=ha, va='center' if off[1] == 0 else ('bottom' if off[1] > 0 else 'top'), fontsize=8)
    ax.set_xlim(0.26, 1.045);ax.set_xticks([0.4, 0.6, 0.8, 1.0])


ALLNUM = None
QWEN_FIG = False


def draw_qwen(ax, metric, scale, label_euler):
    """Qwen-Drive native-horizon curve, paired bootstrap band, and registered samplers including E6."""
    import json
    h = json.loads((fs.ART / QV3 / 'summary.json').read_text())['horizons']['native_5s']
    c = h['curve'];taus = sorted((float(k.split('_t')[1]) for k in c), reverse=True)
    base = c['heun32_t1.0'][metric]
    y = np.array([(c[f'heun32_t{t}'][metric] - base) * scale for t in taus])
    band = json.loads((fs.ART / QBAND / 'summary.json').read_text())['curve'][metric]  # paired to tau=1, as figstyle.curve
    lo = np.array([band[str(t)]['lo'] for t in taus]) * scale;hi = np.array([band[str(t)]['hi'] for t in taus]) * scale
    assert np.allclose([band[str(t)]['point'] * scale for t in taus], y)
    ax.axhline(0, color=fs.GRAY, lw=0.4, ls=':', zorder=0)
    ax.fill_between(taus, lo, hi, color=fs.BAND, alpha=0.6, lw=0, zorder=1)
    ax.plot(taus, y, color=fs.BLACK, lw=1.0, zorder=2)
    draw_margin(ax, taus, y, metric, scale)
    smp = dict(h['samplers'])
    if (fs.ART / QE6 / 'summary.json').exists():
        q6 = json.loads((fs.ART / QE6 / 'summary.json').read_text());assert q6['R2']['pass']
        smp['euler6_t1.0'] = q6['horizons']['native_5s']['samplers']['euler6_t1.0']
    for k in (10, 6, 4, 2, 1):
        if f'euler{k}_t1.0' not in smp:
            continue
        e = smp[f'euler{k}_t1.0'];assert e[metric]['reference'] == base
        if not e['in_range']:  # below the grid: left edge at the coldest grid point (truncated tau_eff = 0.3)
            assert e['below_grid'] and abs(e['tau_eff'] - min(taus)) < 1e-9
            v = (e[metric]['sampler'] - base) * scale
            draw_below(ax, min(taus), v, f'$E_{{{k}}}$')
            print(f"qwendrive  euler{k}_t1.0 below the grid; drawn at tau={min(taus):.1f}, {metric} d{v:+.4f} (no residual)");continue
        x, lo, hi = e['tau_eff'], *e['tau_eff_ci95']
        v = (e[metric]['sampler'] - base) * scale;m = (e[metric]['matched'] - base) * scale
        ax.plot([x, x], [m, v], color='#9CA3AA', lw=0.55, ls=(0, (1.5, 1.7)), zorder=2, solid_capstyle='butt')
        ax.errorbar(x, v, xerr=[[x - lo], [hi - x]], fmt='o', ms=4.0, color=fs.BLUE, mec='white', mew=0.4,
                    elinewidth=0.65, capsize=0, zorder=4)
        if label_euler:
            off = ({4: (5, 1), 6: (0, 6), 10: (-5, 5)} if metric == 'ade1' else {4: (4, -4), 6: (0, 5), 10: (-5, 5)})[k]
            ax.annotate(f'$E_{{{k}}}$', (x, v), xytext=off, textcoords='offset points', ha='center' if off[0] == 0 else ('left' if off[0] > 0 else 'right'),
                        va='bottom' if off[1] > 0 else ('top' if off[1] < 0 else 'center'), fontsize=8)
        print(f"qwendrive  euler{k}_t1.0 tau {x:.2f} [{lo:.2f},{hi:.2f}] {metric} d{v:+.4f}/res{v - m:+.4f}")
    hs = json.loads((fs.ART / QHEUN / 'summary.json').read_text())['horizons']['native_5s']['samplers']
    for k in (3, 5):  # same rule as the Alpamayo columns: Heun-k only when tau_eff is inside the curve's range
        e = hs[f'heun{k}_t1.0']
        if not e['in_range']:
            assert e['above_grid'] and e[metric]['matched'] == base
            v = (e[metric]['sampler'] - base) * scale
            draw_above(ax, v, 0., None, (0, 0))
            if metric == 'flip' and k == 3 and QWEN_FIG:  # label near the triangles, no leader
                ax.annotate('Heun-3, 5', (1., v), xytext=(-8, 12), textcoords='offset points', ha='right', va='bottom', fontsize=7, color=fs.VERMILLION, arrowprops=dict(arrowstyle='-', color=fs.VERMILLION, lw=0.45, shrinkA=2, shrinkB=3))  # just above-left of the triangles, clear of the band
            print(f"qwendrive  heun{k}_t1.0 above the grid (spread {e['spread']:.4f}); drawn at tau=1, {metric} d{v:+.4f}");continue
        x, lo, hi = e['tau_eff'], *e['tau_eff_ci95'];assert e[metric]['reference'] == base
        v = (e[metric]['sampler'] - base) * scale;m = (e[metric]['matched'] - base) * scale
        ax.plot([x, x], [m, v], color='#9CA3AA', lw=0.55, ls=(0, (1.5, 1.7)), zorder=2, solid_capstyle='butt')
        ax.errorbar(x, v, xerr=[[x - lo], [hi - x]], fmt='D', ms=3.0, color=fs.VERMILLION, mec='white', mew=0.4,
                    elinewidth=0.65, capsize=0, zorder=4)
        if metric == 'flip':
            ax.annotate(f'Heun-{k}', (x, v), xytext=(-4, 3), textcoords='offset points', ha='right', va='bottom', fontsize=8)
    ax.set_xlim(0.26, 1.045);ax.set_xticks([0.4, 0.6, 0.8, 1.0])


def legend_handles(below=True):
    h = [plt.Line2D([], [], color=fs.BLACK, lw=1.0, label=r'reference $A_\tau$'),
         plt.Line2D([], [], color=fs.BLUE, lw=0, marker='o', ms=3.4, label=r'$E_k$ at $\tau_{\mathrm{eff}}$'),
         plt.Line2D([], [], color=fs.VERMILLION, lw=0, marker='D', ms=3, label=r'Heun-$k$ at $\tau_{\mathrm{eff}}$'),
         plt.Line2D([], [], color=fs.BLUE, lw=0, marker='<', ms=3.2, label=r'$E_k$ below grid'),
         plt.Line2D([], [], color=fs.VERMILLION, lw=0, marker='>', ms=3.6, label=r'Heun-$k$ above grid'),
         plt.Line2D([], [], color=fs.GRAY, lw=0.8, label='residual'),
         plt.Line2D([], [], color=fs.GRAY, lw=0.7, ls='--', label='margin')]
    return h


QYLABEL = {'ade1': '$\\Delta$ADE$_1$\nvs. $A_1$ (m)', 'flip': 'decisions\nchanged (%)'}  # same text as Fig. 4, two lines


def main_qwen(args):
    """v7.6: Qwen-Drive alone, single column, (a) ADE1 and (b) flips side by side, own y scales. No legend (symbols as in
    Fig. 4). Drawn at 2.9 in so that, placed at \\columnwidth, fonts and lines print at the same size as Fig. 4 at
    \\textwidth (Fig. 4's tight PDF is scaled by ~1.17 when printed)."""
    global QWEN_FIG
    QWEN_FIG = True
    fs.style(big=True)
    fig, axes = plt.subplots(1, 2, figsize=(2.81, 1.0), gridspec_kw=dict(wspace=0.5))
    for ax, metric, letter in zip(axes, ('ade1', 'flip'), 'ab'):
        draw_qwen(ax, metric, ROWS[metric][1], label_euler=True)
        ax.set_ylabel(QYLABEL[metric], linespacing=1.0);ax.set_xlabel(r'temperature $\tau$ or $\tau_{\mathrm{eff}}$')
        fs.panel_letter(ax, letter, x=-0.30, y=1.04);finish_panel(ax, metric)
    fs.save(fig, args.out)


def main():
    global ALLNUM
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--rows', default='ade1,flip,minade6')
    p.add_argument('--out', default='fig_curves.pdf');p.add_argument('--qwen', action='store_true', help='Qwen-Drive single-column figure')
    args = p.parse_args()
    if args.qwen:
        return main_qwen(args)
    rows = args.rows.split(',')
    global QWEN_FIG
    QWEN_FIG = True  # Heun-3/5 label with a leader in the Qwen-Drive column
    fs.style(big=True)
    names, T = fs.tables();samples = fs.boot_samples(names);num = fs.numbers()['models'];ALLNUM = num
    from matplotlib.legend_handler import HandlerTuple
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.40 * len(rows) + 0.55), squeeze=False,
                             gridspec_kw=dict(hspace=0.32, wspace=0.40))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.81)
    for j, model in enumerate(MODELS):
        axes[0, j].set_title(fs.LABEL[model], fontsize=9, pad=8)
        for i, metric in enumerate(rows):
            label, scale = ROWS[metric]
            draw(axes[i, j], T[model], names, samples, num[model], model, metric, scale,
                 label_euler=(i == 0), label_heun=(metric == 'flip'))
            if j == 0:
                axes[i, j].set_ylabel(label, labelpad=5)
    for i, metric in enumerate(rows):
        finish_row(list(axes[i, :3]), metric)
        if metric == 'minade6':
            axes[i, 3].axis('off');continue
        draw_qwen(axes[i, 3], metric, ROWS[metric][1], label_euler=True)
        finish_panel(axes[i, 3], metric, QFIXED.get(metric))
        axes[i, 3].set_yticks({'ade1': [-0.05, 0, 0.05], 'flip': [0, 1, 2, 3, 4]}[metric])
        for j, ax in enumerate(axes[i]):
            if metric == 'minade6' and j == 3:
                continue
            ax.tick_params(axis='both', labelsize=7, length=2.7, width=0.6)
            ax.spines['left'].set_linewidth(0.6);ax.spines['bottom'].set_linewidth(0.6)
            if i < len(rows) - 1:
                ax.tick_params(labelbottom=False)
        axes[i, 0].text(-0.30, 1.04, '(' + 'abcdefg'[i] + ')', transform=axes[i, 0].transAxes,
                        fontsize=9, fontweight='bold', ha='left', va='bottom')
    axes[0, 3].set_title('Qwen-Drive', fontsize=9, pad=8)
    # Six short keys in two rows; the interval sample is an actual errorbar glyph.
    interval = axes[0, 0].errorbar([], [], xerr=[], fmt='o', color=fs.BLUE, ms=3.4,
                                  elinewidth=0.65, capsize=0)
    outside = (plt.Line2D([], [], color=fs.BLUE, marker='<', ms=3.4, lw=0),
               plt.Line2D([], [], color=fs.VERMILLION, marker='>', ms=3.4, lw=0))
    handles = [plt.Line2D([], [], color=fs.BLACK, lw=1.0), interval,
               plt.Line2D([], [], color=fs.VERMILLION, marker='D', ms=3.1, lw=0),
               plt.Line2D([], [], color='#9CA3AA', lw=0.55, ls=(0, (1.5, 1.7))),
               plt.Line2D([], [], color='#A6ABB0', lw=0.55, ls=(0, (3, 2.4))), outside]
    labels = [r'reference $A_\tau$ (95% CI)', r'Euler $E_k$ (95% $\tau$ CI)', 'Heun',
              'residual', 'equivalence bounds', 'outside reference grid']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1.04), ncol=3,
               fontsize=7.5, handlelength=1.8, columnspacing=1.8, handletextpad=0.55,
               labelspacing=0.65, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
    fig.supxlabel(r'reference temperature $\tau$ or effective temperature $\tau_{\mathrm{eff}}$',
                  fontsize=8.5, x=0.54, y=0.015)
    fs.save(fig, args.out)
    for model in MODELS:
        for c, e in num[model].items():
            if e['in_range'] and c.endswith('_t1.0'):
                print(f"{model:11s} {c:13s} tau {e['tau_eff'][0]:.2f} " + ' '.join(
                    f"{m} d{(e[m]['sampler'] - e[m]['reference']) * ROWS[m][1]:+.2f}/res{(e[m]['sampler'] - e[m]['matched']) * ROWS[m][1]:+.2f}" for m in rows))


if __name__ == '__main__':
    main()
