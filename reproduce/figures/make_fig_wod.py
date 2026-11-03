"""Long-tail figure (column width): rater score against temperature on WOD-E2E (Alpamayo-1.5). v6.2: two stacked
panels, (a) the registered primary 280 segments and (b) the registered secondary analysis of all 479 rated frames;
E_1 (below the grid, residual far outside the axis) is noted by an arrow instead of a full-height bar; a residual bar
that leaves the band is drawn in vermillion.

Black curve: the reference (Heun-32) started from tau*z, change in rater score from tau=1 (the 95% segment-bootstrap
band of the curve is printed, not drawn: it is wider than the paired residual intervals and would hide them). Light band: the registered equivalence margin
(+-0.08) around the curve. Blue markers: E_10, E_4 at their effective temperature (horizontal bar: 95% interval of
tau_eff); E_1 lies below the temperature grid (tau_eff <= 0.1) and is drawn at 0.1 with a left arrow. Vertical bar:
the 90% interval of the residual (10 000 segment resamples, from the artifact), drawn around the curve value at
tau_eff, so a bar inside the light band is an equivalence pass.
Data: artifacts/wod_e2e_analysis_primary_converged10k_v2_20261029/{summary,per_segment}.json.
v6.3a: the score is the official Waymo rater feedback score (unmodified official module) on the same plans, from
artifacts/wod_e2e_analysis_official_rfs_v1_20261030/{summary,per_segment}.json; the primary sample is the first 280
segments in name order, as registered.

Run from the repo root:  python reproduce/figures/make_fig_wod.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

OFF = fs.ART / 'wod_e2e_analysis_official_rfs_v1_20261030'  # v6.3a: official RFS, both samples in one artifact
TAUS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def panel(ax, sample, title):
    O = json.loads((OFF / 'summary.json').read_text());S = O['samples'][sample];P = json.loads((OFF / 'per_segment.json').read_text())['per_segment']
    names = sorted(P)[:S['n_segments']];margin = O['margins']['rfs']
    arr = np.array([[P[n][f'heun32_t{t}']['rfs_official'] - P[n]['heun32_t1.0']['rfs_official'] for t in TAUS] for n in names])
    y = arr.mean(axis=0)
    # curve point estimate must equal the artifact's configuration means
    assert np.allclose(y, [S['configs'][f'heun32_t{t}']['rfs'] - S['configs']['heun32_t1.0']['rfs'] for t in TAUS])
    base = S['configs']['heun32_t1.0']['rfs'];t = np.array(TAUS)
    ax.plot(t, y + margin, color=fs.GRAY, lw=0.7, ls='--', zorder=1, label='margin ($\\pm$0.08)')  # dashed, not gray fill
    ax.plot(t, y - margin, color=fs.GRAY, lw=0.7, ls='--', zorder=1)
    ax.plot(t, y, color=fs.BLACK, lw=1.0, zorder=2, label='reference at $\\tau$')
    ax.axhline(0, color=fs.GRAY, lw=0.4, ls=':', zorder=0)
    YLO = -0.06
    for k, name, off, ha in (('euler10_t1.0', '$E_{10}$', (-6, -13), 'right'), ('euler4_t1.0', '$E_4$', (-6, -13), 'right')):
        s = S['samplers'][k];r = s['rfs'];x = s['tau_eff'];v = r['sampler'] - base;m = r['matched'] - base
        rlo, rhi = r['residual_ci90']
        ok = r['equivalent']
        ax.plot([x, x], [m + rlo, m + rhi], color=fs.EULER[1] if ok else fs.VERMILLION, lw=2.0, alpha=0.9, solid_capstyle='butt', zorder=3)
        tl, th = s['tau_eff_ci95']
        ax.errorbar(x, v, xerr=[[x - tl], [th - x]], fmt='o', ms=5, color=fs.BLUE, mec='white', mew=0.6, elinewidth=0.8, capsize=0, zorder=4)
        ax.annotate(name + (' passes' if ok else ' misses'), (x, v), xytext=off, textcoords='offset points', ha=ha, va='center', fontsize=8.5)
    e1 = S['samplers']['euler1_t1.0'];assert not e1['in_range']
    r1 = e1['rfs']['residual']
    ax.annotate(f'$E_1$ ($\\tau_{{\\mathrm{{eff}}}}\\leq0.1$): residual ${r1:.2f}$, below the axis', (0.1, YLO + 0.01), xytext=(0.1, YLO + 0.075),
                ha='left', va='center', fontsize=7.5, color='#555555', arrowprops=dict(arrowstyle='->', lw=0.6, color=fs.GRAY))
    ax.set_xlim(0.05, 1.03);ax.set_ylim(YLO, 0.36);ax.set_xticks([0.1, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('$\\Delta$ rater score')
    ax.set_title(title % S['n_segments'], fontsize=8.3, loc='left')
    return S


def main():
    fs.style(big=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(fs.COL, 3.55), sharex=True, gridspec_kw=dict(hspace=0.32))
    panel(a, '280', '(a) registered primary: first %d segments')
    panel(b, '479', '(b) registered secondary: all %d rated frames')
    b.set_xlabel('temperature $\\tau$ or $\\tau_{\\mathrm{eff}}$')
    a.legend(loc='upper right', fontsize=7.5, handlelength=1.4)
    fs.save(fig, 'fig_wod.pdf')


if __name__ == '__main__':
    main()
