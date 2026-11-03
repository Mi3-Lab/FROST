"""Figure 1 (teaser, single column).

(a) One PhysicalAI-AV test clip, Alpamayo-1.5, six shared noise seeds: the reference solver A_1
    (black) and four Euler steps E_4 (blue) from the same noises. Gray arrows join the two
    endpoints of each seed; E_4 endpoints move toward the one-step mean E_1 (star).
(b) Decision flips vs temperature, Alpamayo-1.5, 60 clips x 6 seeds: the explicit-temperature
    curve A_tau (black, 95% clip-bootstrap band), Euler samplers placed at their effective
    temperature (blue), Heun-3/Heun-5 placed at theirs (vermillion), far above the curve.

Run from the repo root:  python reproduce/figures/make_fig_teaser.py [--clip context_101]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from frost.core.decisions import decisions

MODEL = 'alpamayo15'


def bev(ax, clip):
    P = fs.plans(MODEL)
    rec = {r['name']: r for r in json.loads((fs.DATA / 'records.json').read_text()) if 'error' not in r}
    speed = rec[clip]['speed_mps']
    get = lambda c: [P[f'{clip}_s{s}_{c}'][:, :2] for s in fs.SEEDS]
    ref, e4, e1 = get('heun32_t1.0'), get('euler4_t1.0'), get('euler1_t1.0')
    mean = np.mean(e1, 0)  # identical for every seed up to numerics
    key = lambda t: tuple(decisions(np.c_[t, np.zeros(len(t))], speed)[k] for k in ('longitudinal', 'lateral'))
    flips = sum(key(a) != key(b) for a, b in zip(ref, e4))
    X = lambda t: -t[:, 1]  # lateral, left of ego on the left
    Y = lambda t: t[:, 0]   # longitudinal, forward is up
    for t in ref:
        ax.plot(X(t), Y(t), color=fs.BLACK, lw=0.7, zorder=2)
    for t in e4:
        ax.plot(X(t), Y(t), color=fs.BLUE, lw=0.9, zorder=3)
    for a, b in zip(ref, e4):
        ax.annotate('', xy=(X(b)[-1], Y(b)[-1]), xytext=(X(a)[-1], Y(a)[-1]),
                    arrowprops=dict(arrowstyle='-|>', color=fs.GRAY, lw=0.6, mutation_scale=6, shrinkA=2, shrinkB=2), zorder=4)
    ax.plot(X(mean), Y(mean), color=fs.EULER[1], lw=0.8, ls=(0, (2, 1.5)), zorder=3)
    ax.plot(X(mean)[-1], Y(mean)[-1], marker='*', ms=8, color=fs.EULER[1], mec='white', mew=0.5, zorder=6, ls='none')
    ax.plot([X(t)[-1] for t in ref], [Y(t)[-1] for t in ref], 'o', ms=3.2, mfc='white', mec=fs.BLACK, mew=0.7, zorder=5, ls='none')
    ax.plot([X(t)[-1] for t in e4], [Y(t)[-1] for t in e4], 'o', ms=3.2, color=fs.BLUE, mec='white', mew=0.4, zorder=5, ls='none')
    ax.plot(0, 0, marker=(3, 0, 0), ms=6, color=fs.GRAY, zorder=6, ls='none')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('lateral (m)')
    ax.set_ylabel('longitudinal (m)')
    ax.text(X(mean)[-1] + 2, Y(mean)[-1] + 1, '$E_1$', ha='left', va='bottom', fontsize=7)
    far = int(np.argmax([np.linalg.norm(a[-1] - b[-1]) for a, b in zip(ref, e4)]))  # the seed that moves most
    ax.text(X(e4[far])[-1] - 3, Y(e4[far])[-1], '$E_4$', ha='right', va='center', fontsize=7)
    ax.text(X(ref[far])[-1] - 3, Y(ref[far])[-1] + 2, '$A_1$', ha='right', va='bottom', fontsize=7)
    return flips, speed


def flips_vs_tau(ax):
    names, T = fs.tables()
    table = T[MODEL];samples = fs.boot_samples(names);num = fs.numbers()['models'][MODEL]
    tau, y, lo, hi = fs.curve(table, names, 'flip', samples, 100)
    ax.fill_between(tau, lo, hi, color=fs.BAND, alpha=0.6, lw=0, zorder=1)
    ax.plot(tau, y, color=fs.BLACK, lw=1.0, zorder=2)
    xl = 0.55; ax.text(xl, np.interp(xl, *zip(*sorted(zip(tau, lo)))) - 0.4, r'$A_\tau$', ha='center', va='top', fontsize=7)  # under the band
    out = {}
    for k in [2, 4, 6, 10]:
        e = num[f'euler{k}_t1.0']
        if not e['in_range']:
            continue
        x = e['tau_eff'];v, vlo, vhi = fs.sampler_ci(table, names, f'euler{k}_t1.0', 'flip', samples, 100)
        ax.errorbar(x[0], v, xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[v - vlo], [vhi - v]], fmt='o', ms=3.6,
                    color=fs.BLUE, mec='white', mew=0.4, elinewidth=0.6, capsize=0, zorder=4)
        dx, dy = {2: (4, 4), 4: (4, 4), 6: (4, 4), 10: (4, 4)}[k]
        ax.annotate(f'$E_{{{k}}}$', (x[0], v), xytext=(dx, dy), textcoords='offset points', ha='left', va='bottom',
                    fontsize=7, color=fs.BLACK)
        out[k] = (x[0], v)
    for k, dy in [(3, 0), (5, 0)]:
        e = num[f'heun{k}_t1.0'];x = e['tau_eff'];v, vlo, vhi = fs.sampler_ci(table, names, f'heun{k}_t1.0', 'flip', samples, 100)
        ax.errorbar(x[0], v, xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[v - vlo], [vhi - v]], fmt='D', ms=3.4,
                    color=fs.VERMILLION, mec='white', mew=0.4, elinewidth=0.6, capsize=0, zorder=4)
        ax.annotate(f'Heun-{k}', (x[0], v), xytext=(-5, 4), textcoords='offset points', ha='right', va='bottom', fontsize=7)
        out[f'heun{k}'] = (x[0], v)
    ax.set_xlim(0.37, 1.03)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'temperature $\tau$ or $\tau_{\mathrm{eff}}$')
    ax.set_ylabel(r'decisions changed (%)')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--clip', default='context_101')
    p.add_argument('--out', default='fig_teaser.pdf');args = p.parse_args()
    fs.style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.COL, 1.9), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.45))
    flips, speed = bev(a, args.clip)
    pts = flips_vs_tau(b)
    fs.panel_letter(a, 'a', x=-0.30);fs.panel_letter(b, 'b', x=-0.25)
    h = [plt.Line2D([], [], color=fs.BLACK, lw=0.9, marker='o', mfc='white', ms=3, label=r'reference $A_1$ / $A_\tau$'),
         plt.Line2D([], [], color=fs.BLUE, lw=0.9, marker='o', ms=3, label=r'Euler $E_k$'),
         plt.Line2D([], [], color=fs.EULER[1], lw=0, marker='*', ms=7, label=r'$E_1$ (cond. mean)'),
         plt.Line2D([], [], color=fs.VERMILLION, lw=0, marker='D', ms=3, label='Heun')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=4, handlelength=1.3, columnspacing=0.8,
               handletextpad=0.4, borderaxespad=0)
    fs.save(fig, args.out)
    print(f'clip {args.clip}: ego {speed:.1f} m/s, E4 changes {flips}/6 decisions vs A1; markers', pts)


if __name__ == '__main__':
    main()
