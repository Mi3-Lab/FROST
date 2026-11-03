"""Page-one overview: observation, matched-temperature method, decision evidence.

(a) Original fixed context_101, all six shared noises: the reference and four
Euler steps, with the one-step mean. No exemplar or noise was re-selected.
(b) Reference spread on the 400 held-out clips, with E4/E10 matching guides.
(c) The original reference flip-rate curve, Euler E2/E4/E6/E10 and Heun3/5,
with all original intervals and in-range filtering preserved.

CPU reproduction from repo root:
MPLBACKEND=Agg python reproduce/figures/make_fig_teaser_wide.py
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from make_fig_teaser import bev, flips_vs_tau
from make_fig_test import spread_panel


def gains(ax):
    models = list(fs.LABEL);num = fs.numbers()['models'];x = np.arange(len(models));w = 0.36
    for i, m in enumerate(models):
        a = num[m]['euler4_t1.0']['ade1'];eff = -np.array(a['effect'])
        ax.bar(i - w / 2, eff[0], w, color=fs.BLUE, yerr=[[eff[0] - eff[2]], [eff[1] - eff[0]]],
               error_kw=dict(elinewidth=0.6, capsize=0, ecolor=fs.BLACK))
        ax.bar(i + w / 2, a['reference'] - a['matched'], w, color=fs.BAND, edgecolor=fs.BLACK, lw=0.6)
    ax.set_xticks(x, [fs.LABEL[m].replace('Alpamayo-', 'A-') for m in models])
    ax.set_ylabel(r'ADE$_1$ gain over $A_1$ (m)')
    ax.set_ylim(0, 0.52)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=fs.BLUE, label='4 Euler steps'),
                       plt.Rectangle((0, 0), 1, 1, fc=fs.BAND, ec=fs.BLACK, lw=0.6, label=r'$A_1$ run cooler ($\tau_{\mathrm{eff}}$)')],
              loc='upper left', bbox_to_anchor=(0, 1.12), handlelength=1.0, handletextpad=0.4)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--clip', default='context_101')
    p.add_argument('--out', default='fig_teaser_wide.pdf');args = p.parse_args()
    fs.style(big=True)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(fs.TEXT, 1.68),
            gridspec_kw=dict(width_ratios=[0.9, 1, 1.12], wspace=0.48))
    fs.use('orig');bev(a, args.clip);fs.use('pooled')
    spread_panel(b, compact=True);flips_vs_tau(c)
    a.set_title('(a) Fewer steps contract plans', loc='left', fontsize=8.5)
    b.set_title('(b) Match the spread', loc='left', fontsize=8.5)
    c.set_title('(c) Compare decision rates', loc='left', fontsize=8.5)
    for ax, l in zip((a, b, c), 'abc'):
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(7.5);ax.yaxis.label.set_size(7.5)
    b.set_ylabel('spread (m)')
    b.set_xlabel(r'reference temperature $\tau$')
    c.set_xlabel(r'$\tau$ or $\tau_{\mathrm{eff}}$')
    h = [plt.Line2D([], [], color=fs.BLACK, lw=0.9, marker='o', mfc='white', ms=3, label=r'reference $A_\tau$'),
         plt.Line2D([], [], color=fs.BLUE, lw=0.9, marker='o', ms=3, label=r'Euler $E_k$'),
         plt.Line2D([], [], color=fs.EULER[1], lw=0, marker='*', ms=6, label=r'one-step proxy $E_1$'),
         plt.Line2D([], [], color=fs.VERMILLION, lw=0, marker='D', ms=3, label='Heun')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.14), ncol=4,
               handlelength=1.4, columnspacing=1.5, handletextpad=0.4, borderaxespad=0, fontsize=8)
    fs.save(fig, args.out)


if __name__ == '__main__':
    main()
