"""Mechanism figure (single column): Euler contraction from the Gaussian factor of Eq. (1).

(a) Closed form, one coordinate of a Gaussian target N(m, s^2) with s = 0.5, rectified flow.
    Deviation from the mean path, |x_t - t m| / |z|, for the exact flow (black: ||u(t)||) and for
    Euler with k uniform steps (blue, darker = fewer steps). Each Euler step multiplies the
    deviation by cos(theta_i) relative to the exact flow, so E_k ends at s * prod_i cos(theta_i);
    E_1 ends at the mean.
(b) Measured median contraction ratio |S(z) - E_1| / |A_1(z) - E_1| in trajectory space, 400 held-out clips x
    6 seeds per model (the pooled artifact selected by figstyle), against field evaluations (NFE).
    Gray band: Gaussian factor prod_i cos(theta_i) for s in [0.3, 1] (Eq. 1, no fitted parameters).
    Hollow squares: Gaussian prediction from Alpamayo-1.5's conditional action variance
    (median and IQR, artifacts/gaussian_contraction_v1_20261026; 200 clips, action space).
    Heun-k (2k NFE) in vermillion barely contracts at the same budget.

Run from the repo root:  python reproduce/figures/make_fig_mechanism.py
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs


def u_norm(t, s):
    return np.sqrt((1 - t) ** 2 + (t * s) ** 2)


def euler_path(k, s):
    """Deviation from the mean path under k uniform Euler steps, normalized by |z|."""
    t = np.linspace(0, 1, k + 1);d = [1.0]
    for i in range(k):
        a, b = np.array([1 - t[i], t[i] * s]), np.array([1 - t[i + 1], t[i + 1] * s])
        d.append(d[-1] * a @ b / (a @ a))  # Euler multiplier <u_i, u_{i+1}> / |u_i|^2
    return t, np.array(d)


def factor(k, s):
    """Gaussian factor prod_i cos(theta_i) of Eq. (1)."""
    t, d = euler_path(k, s)
    return d[-1] / u_norm(1., s)


def panel_paths(ax, s=0.5):
    t = np.linspace(0, 1, 200)
    ax.plot(t, u_norm(t, s), color=fs.BLACK, lw=1.1, zorder=3)
    ax.text(0.72, u_norm(0.72, s) + 0.06, 'exact', ha='center', va='bottom', fontsize=8)
    for k in [1, 2, 4, 10]:
        tk, d = euler_path(k, s)
        ax.plot(tk, d, color=fs.EULER[k], lw=0.9, marker='o', ms=1.8 if k == 10 else 2.4, zorder=2)
        f = factor(k, s)
        ax.annotate(f'$E_{{{k}}}$', (1, d[-1]), xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=8)
    ax.plot(1, s, 'o', ms=2.8, color=fs.BLACK, zorder=4)
    ax.set_xlim(0, 1);ax.set_ylim(-0.03, 1.05);ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel('flow time $t$')
    ax.set_ylabel(r'$|x_t - t\,m|\,/\,|z|$')
    ax.set_title(f'Gaussian target ($s={s}$)', fontsize=8)


def panel_measured(ax):
    num = fs.numbers()['models'];g = fs.gauss()
    ks = np.arange(1, 11)
    lo = [min(factor(k, s) for s in np.linspace(0.3, 1, 30)) for k in ks]
    hi = [max(factor(k, s) for s in np.linspace(0.3, 1, 30)) for k in ks]
    ax.fill_between(ks, lo, hi, color=fs.BAND, alpha=0.6, lw=0, zorder=1)
    ax.axhline(1, color=fs.GRAY, lw=0.5, ls=':', zorder=0)
    for k, e in g['steps'].items():
        k = int(k);m = e['predicted_median'];q = e['predicted_iqr']
        ax.errorbar(k + 0.25, m, yerr=[[m - q[0]], [q[1] - m]], fmt='s', ms=3, mfc='white', mec=fs.GRAY, color=fs.GRAY,
                    mew=0.7, elinewidth=0.6, capsize=0, zorder=3)
    jitter = {'alpamayo15': -0.2, 'alpamayo_r1': 0.0, 'alpamayo2': 0.2}
    for model, v in num.items():
        xs = [k for k in (1, 2, 4, 6, 10)];ys = [v[f'euler{k}_t1.0']['contraction']['median'] for k in xs]
        ax.plot(np.array(xs) + jitter[model], ys, color=fs.BLUE, lw=0.5, alpha=0.6, zorder=2)
        ax.plot(np.array(xs) + jitter[model], ys, fs.MARKER[model], ms=3.2, color=fs.BLUE, mec='white', mew=0.4, ls='none', zorder=4)
    for k in (3, 5):
        e = num['alpamayo15'].get(f'heun{k}_t1.0')
        if e:
            ax.plot(2 * k, e['contraction']['median'], 'D', ms=3.2, color=fs.VERMILLION, mec='white', mew=0.4, zorder=5)
            ax.annotate(f'Heun-{k}', (2 * k, e['contraction']['median']), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax.set_xlim(0.4, 10.8);ax.set_ylim(-0.03, 1.12);ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel('field evaluations (NFE)')
    ax.set_ylabel('contraction ratio')
    h = [plt.Line2D([], [], color=fs.BLUE, marker=fs.MARKER[m], ms=3, lw=0, label=fs.LABEL[m].replace('Alpamayo-', 'A-')) for m in fs.MARKER]
    h += [plt.Line2D([], [], color=fs.GRAY, marker='s', mfc='white', ms=3, lw=0, label='Gaussian prediction')]
    ax.legend(handles=h, loc='lower right', handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1, fontsize=6.7)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', default='fig_mechanism.pdf');args = p.parse_args()
    fs.style(big=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.COL, 2.1), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.85))
    panel_paths(a);panel_measured(b)
    b.set_title('Measured contraction', fontsize=8)
    fs.panel_letter(a, 'a', x=-0.3);fs.panel_letter(b, 'b', x=-0.3)
    fs.save(fig, args.out)
    print('Gaussian factor at s=0.5:', {k: round(factor(k, .5), 3) for k in (1, 2, 4, 10)},
          '| s in [0.3,1], k=10:', round(min(factor(10, s) for s in np.linspace(.3, 1, 30)), 3), '-',
          round(max(factor(10, s) for s in np.linspace(.3, 1, 30)), 3))


if __name__ == '__main__':
    main()
