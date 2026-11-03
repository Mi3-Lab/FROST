"""Schematic of the matched-temperature test (text width since v6.2, larger annotations), drawn on real data: Alpamayo-1.5, 400 held-out clips.

(a) Spread of the reference (Heun-32 from tau*z) against tau; the spread of E_10 and E_4 is read off the curve to give
    their effective temperature (dashed guides).
(b) Decision flips against A_1 of the reference at tau (curve) and of each sampler placed at its tau_eff; the
    residual is the vertical distance to the curve (bracket), judged against the margin band (+-2 points).
    Heun-5 sits at tau_eff ~ 1 and far above the curve.
Per-clip metrics come from the paper's own generator (figstyle.tables); tau_eff and the matched values from
artifacts/paper_numbers_heldout_pooled400_v2_20261028/summary.json.

Run from the repo root:  python reproduce/figures/make_fig_test.py
"""
import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs


def spread_panel(a, compact=False):
    names, T = fs.tables();t = T['alpamayo15'];num = fs.numbers()['models']['alpamayo15']
    taus = np.array(fs.TAUS)
    spr = np.array([np.mean([t[f'heun32_t{x}'][n]['spread'] for n in names]) for x in fs.TAUS])
    flp = 100 * np.array([np.mean([t[f'heun32_t{x}'][n]['flip'] for n in names]) for x in fs.TAUS])
    # (a) read tau_eff off the spread curve
    a.plot(taus, spr, color=fs.BLACK, lw=1.0)
    for k, lab in ((10, '$E_{10}$'), (4, '$E_4$')):
        s = np.mean([t[f'euler{k}_t1.0'][n]['spread'] for n in names]);x = num[f'euler{k}_t1.0']['tau_eff'][0]
        a.plot([0.37, x], [s, s], ls='--', lw=0.6, color=fs.EULER[k]);a.plot([x, x], [s, 0.45], ls='--', lw=0.6, color=fs.EULER[k])
        a.plot(x, s, 'o', ms=5, color=fs.EULER[k], mec='white', mew=0.4, zorder=4)
        a.annotate(lab, (0.39, s), xytext=(0, 2), textcoords='offset points', fontsize=8, va='bottom')
        if compact:
            a.annotate(f'{x:.2f}', (x, 0.45), xytext=(0, 3), textcoords='offset points', fontsize=7.5, va='bottom', ha='center', color=fs.EULER[k])
        else:
            a.annotate(f'$\\tau_{{\\mathrm{{eff}}}}={x:.2f}$', (x, 0.45), xytext=(-2, 2), textcoords='offset points', fontsize=7.5, va='bottom', ha='right', color=fs.EULER[k])
    a.set_xlim(0.37, 1.03);a.set_ylim(0.45, 1.85);a.set_xticks([0.4, 0.6, 0.8, 1.0])
    a.set_xlabel('reference temperature $\\tau$');a.set_ylabel('spread $\\mathrm{Spr}$ (m)')
    a.set_title('Match the spread', loc='left')
    return taus, flp, num


def main():
    fs.style(big=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.TEXT, 2.0), gridspec_kw=dict(wspace=0.27))
    taus, flp, num = spread_panel(a)
    # (b) residual against the curve at tau_eff
    for d in (-2, 2):  # margin as dashed lines; gray fill means a 95% bootstrap band in every figure
        b.plot(taus, flp + d, color=fs.GRAY, lw=0.7, ls='--', zorder=1)
    b.plot(taus, flp, color=fs.BLACK, lw=1.0, zorder=2)
    for key, lab, col, mk in (('euler4_t1.0', '$E_4$', fs.BLUE, 'o'), ('heun5_t1.0', 'Heun-5', fs.VERMILLION, 'D')):
        e = num[key];x = min(e['tau_eff'][0], 1.0);v = 100 * e['flip']['sampler'];m = 100 * e['flip']['matched']
        if v - m > 3:
            b.annotate('', xy=(x, m), xytext=(x, v), arrowprops=dict(arrowstyle='<->', lw=0.6, color=fs.GRAY, shrinkA=2, shrinkB=0))
        b.plot(x, v, mk, ms=5, color=col, mec='white', mew=0.4, zorder=4)
        b.annotate(lab, (x, v), xytext=(-6, 0) if x > 0.95 else (-4, 7), textcoords='offset points', fontsize=9, ha='right', va='center')
    b.annotate('residual', (0.985, 7.5), xytext=(0, 0), textcoords='offset points', fontsize=8.5, ha='right', va='center', color=fs.GRAY)
    b.set_xlim(0.37, 1.05);b.set_ylim(-2.5, 16);b.set_xticks([0.4, 0.6, 0.8, 1.0])
    b.set_xlabel('$\\tau$ or $\\tau_{\\mathrm{eff}}$');b.set_ylabel('decisions changed (%)')
    b.set_title('Test the residual', loc='left')
    fs.panel_letter(a, 'a', x=-0.12, y=1.0);fs.panel_letter(b, 'b', x=-0.12, y=1.0)
    fs.save(fig, 'fig_test.pdf')
    print('flips', np.round(flp, 2))


if __name__ == '__main__':
    main()
