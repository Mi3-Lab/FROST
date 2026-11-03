"""Compact FROST verdict grid for every pair of the complete atlas (v7.1 redesign).

Rows are planners, columns are samplers; each filled cell is one of the 27 model-sampler pairs of the complete
atlas (make_fig_frost_atlas.load(), unchanged JSON verdict rule: every available metric's marginal 90%
equivalence test must pass; this is not a joint test). Empty cells are pairs that were not run. The step-count
columns hold Euler for the flow-matching planners and deterministic DDIM (eta=0) for the diffusion planners.
Every case is a model-sampler pair, not an independent model or replication. No row, failure or out-of-range
result is removed. Run on CPU from the repo root: python reproduce/figures/make_fig_frost_summary.py (prints the grid).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figstyle as fs
from make_fig_frost_atlas import load, TEMP, OWN, NONE

# (row key in the atlas, label, data/sampler family shown in gray)
ROWS = [('alpamayo15', 'Alpamayo-1.5', 'flow'), ('alpamayo_r1', 'Alpamayo-R1', 'flow'), ('alpamayo2', 'Alpamayo-2', 'flow'),
        ('qwen_rl', 'Qwen-Drive', 'flow'), ('flashdrive', 'FlashDrive on Alp.-1.5', 'flow'),
        ('recog_il', 'ReCogDrive, IL', 'diffusion'), ('recog_rl', 'ReCogDrive, RL', 'diffusion'),
        ('diffusiondrive', 'DiffusionDrive', 'diffusion')]
# (column label, sampler keys that land in it)
COLS = [('2', ('euler2_t1.0', 'ddim2_lead_eta0_t1.0')), ('4', ('euler4_t1.0', 'ddim4_lead_eta0_t1.0')),
        ('5', ('ddim5_lead_eta0_t1.0',)), ('6', ('euler6_t1.0',)), ('10', ('euler10_t1.0', 'ddim10_lead_eta0_t1.0')),
        ('20', ('ddim20_lead_eta0_t1.0',)), ('Heun-5', ('heun5_t1.0',)), ('cache-8', ('flashcache8_t1.0',)),
        ('release', ('release5_lead_eta1_t1.0', 'release2_lead_eta0_t1.0'))]
STYLE = {TEMP: (fs.BLUE, 'o', 6.2), OWN: (fs.VERMILLION, 'X', 6.8), NONE: (fs.GRAY, '>', 6.2)}
TXT = {TEMP: 'eq', OWN: 'X', NONE: '>'}


def main():
    fs.style(big=True)
    pairs = {r['key']: r for r in load(None)}
    assert len(pairs) == 27
    grid, used = {}, set()
    for i, (model, _, _) in enumerate(ROWS):
        for j, (_, keys) in enumerate(COLS):
            hit = [k for k in keys if (model, k) in pairs]
            assert len(hit) <= 1
            if hit:
                grid[i, j] = pairs[model, hit[0]]['json'];used.add((model, hit[0]))
    assert used == set(pairs), set(pairs) - used  # every atlas pair is drawn exactly once

    nr, nc = len(ROWS), len(COLS)
    fig = plt.figure(figsize=(fs.TEXT, 1.85))
    ax = fig.add_axes([0.255, 0.02, 0.60, 0.80])
    ax.set_xlim(-0.5, nc - 0.5);ax.set_ylim(nr - 0.5, -0.5);ax.axis('off')
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#F7F7F7' if (i, j) not in grid else 'white',
                                   ec='#DDDDDD', lw=0.5, zorder=0))
    for (i, j), v in grid.items():
        c, m, ms = STYLE[v];ax.plot(j, i, m, color=c, ms=ms, mec='white', mew=0.5, zorder=3)
    gap = 4.5  # flow-matching VLAs (PhysicalAI-AV) above, diffusion planners (NAVSIM) below
    ax.plot([-0.5 - 3.1, nc - 0.5 + 1.7], [gap - 0.0, gap - 0.0], color=fs.GRAY, lw=0.7, clip_on=False, zorder=4)
    for x in (5.5, 6.5, 7.5):
        ax.plot([x, x], [-0.5, nr - 0.5], color='#BBBBBB', lw=0.8, zorder=1)
    # column headers
    for j, (lab, _) in enumerate(COLS):
        ax.text(j, -0.75, lab, ha='center', va='bottom', fontsize=8)
    ax.text(2.5, -1.55, 'steps: Euler (flow) / DDIM $\\eta{=}0$ (diffusion)', ha='center', va='bottom', fontsize=7.5, color=fs.GRAY)
    ax.plot([-0.4, 5.4], [-1.45, -1.45], color=fs.GRAY, lw=0.5, clip_on=False)
    ax.text(7.0, -1.55, 'other samplers', ha='center', va='bottom', fontsize=7.5, color=fs.GRAY)
    ax.plot([5.6, 8.4], [-1.45, -1.45], color=fs.GRAY, lw=0.5, clip_on=False)
    # row labels and right-hand data/sampler family
    for i, (_, lab, fam) in enumerate(ROWS):
        ax.text(-0.75, i, lab, ha='right', va='center', fontsize=8)
    ax.text(nc - 0.5 + 0.25, 2.0, 'flow matching\nPhysicalAI-AV', ha='left', va='center', fontsize=7.2, color=fs.GRAY, linespacing=1.1)
    ax.text(nc - 0.5 + 0.25, 6.0, 'diffusion\nNAVSIM', ha='left', va='center', fontsize=7.2, color=fs.GRAY, linespacing=1.1)
    handles = [Line2D([], [], ls='', marker='o', color=fs.BLUE, ms=5.5, mec='white', mew=0.4, label='equivalent in every tested metric'),
               Line2D([], [], ls='', marker='X', color=fs.VERMILLION, ms=6, mec='white', mew=0.4, label='equivalence not established'),
               Line2D([], [], ls='', marker='>', color=fs.GRAY, ms=5.5, mec='white', mew=0.4, label='outside the temperature grid'),
               Rectangle((0, 0), 1, 1, fc='#F7F7F7', ec='#DDDDDD', lw=0.5, label='not run')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8,
               handletextpad=0.3, columnspacing=1.6)
    fs.save(fig, 'fig_frost_summary.pdf')
    # stdout check: the grid, the per-verdict counts
    w = 9
    print(' ' * 24 + ''.join(f'{c:>{w}s}' for c, _ in COLS))
    for i, (_, lab, _) in enumerate(ROWS):
        print(f'{lab:24s}' + ''.join(f"{TXT[grid[i, j]] if (i, j) in grid else '.':>{w}s}" for j in range(nc)))
    print({v: sum(x == v for x in grid.values()) for v in (TEMP, OWN, NONE)}, 'pairs', len(grid))


if __name__ == '__main__':
    main()
