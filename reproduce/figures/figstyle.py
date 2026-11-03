"""Shared style and data access for the paper figures.

Every number comes from the artifacts the paper already uses:
  - matched-temperature point estimates and intervals: results/paper_numbers_*/summary.json
  - Gaussian contraction prediction: results/gaussian_contraction_v1_20261026/summary.json
  - per-clip metrics (spread, ADE1, minADE6, flips): recomputed with frost.analysis.paper_numbers from the per-clip
    plans ($FROST_ARTIFACTS/temperature_probe_*/plans.npz and the context caches) when those are present, otherwise
    read from results/figure_cache/, which build_figure_cache.py writes from exactly that computation.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro_paths import RESULTS as ART, ARTIFACTS, FIGURES as OUT  # noqa: E402

CACHE = ART / 'figure_cache'
CACHE_METRICS = ['spread', 'ade1', 'minade6', 'flip']
GAUSS = ART / 'gaussian_contraction_v1_20261026' / 'summary.json'
# Two samples. 'pooled' (default, primary evidence since v5.32): 400 held-out clips (context_000-399 of the
# held-out split). 'orig': the original 60 test-split clips (context_100-159), kept for the page-1 example clip.
SAMPLES = {
    'pooled': dict(data=ART / 'physicalai_heldout_contexts_v3_20261027', range=[0, 400],
                   numbers=ART / 'paper_numbers_heldout_pooled400_v2_20261028' / 'summary.json',
                   probes={'alpamayo15': 'temperature_probe_a15_heldout_pooled400_v3_20261028',
                           'alpamayo_r1': 'temperature_probe_r1_heldout_pooled400_v3_20261028',
                           'alpamayo2': 'temperature_probe_a2_heldout_pooled400_v3_20261028'}),
    'orig': dict(data=ART / 'physicalai_test_contexts_v1_20261022', range=[100, 160],
                 numbers=ART / 'paper_numbers_v4_20261026' / 'summary.json',
                 probes={'alpamayo15': 'temperature_probe_v1_20261022', 'alpamayo_r1': 'temperature_probe_r1_v1_20261022',
                         'alpamayo2': 'temperature_probe_a2_v1_20261022'})}
DATA = NUMBERS = PROBES = RANGE = None


def use(sample):
    """Select the clip sample that plans(), tables() and numbers() read."""
    global DATA, NUMBERS, PROBES, RANGE
    c = SAMPLES[sample];DATA, NUMBERS, PROBES, RANGE = c['data'], c['numbers'], c['probes'], c['range']


use('pooled')
LABEL = {'alpamayo15': 'Alpamayo-1.5', 'alpamayo_r1': 'Alpamayo-R1', 'alpamayo2': 'Alpamayo-2'}
MARKER = {'alpamayo15': 'o', 'alpamayo_r1': 's', 'alpamayo2': '^'}
TAUS = [1., .9, .8, .7, .6, .5, .4]
SEEDS = [9201, 9202, 9203, 9204, 9205, 9206]
BOOT, BOOT_SEED = 2000, 20261026

# Okabe-Ito. Reference / explicit temperature = black with a gray band; Euler = blue;
# second-order (Heun) = vermillion; human log = gray dashed.
BLACK, BLUE, SKY, VERMILLION, GREEN, ORANGE = '#000000', '#0072B2', '#56B4E9', '#D55E00', '#009E73', '#E69F00'
BAND, GRAY = '#BBBBBB', '#777777'
# one-hue ramp for Euler step counts (darker = fewer steps = more contraction)
EULER = {1: '#003A5C', 2: '#004F7D', 4: '#0072B2', 6: '#3F95C6', 10: '#4F9CC8'}

COL, TEXT = 3.25, 6.75  # ICML single-column and text widths (inches)


def style(big=False):
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'axes.titleweight': 'normal', 'axes.titlepad': 6, 'axes.labelpad': 3, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0, 'lines.markersize': 4,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.045, 'figure.dpi': 150})
    if big:  # v5.42: larger axis, tick and legend text for the figures that print at column or reduced width
        mpl.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5})


def panel_letter(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontweight='bold', va='bottom', ha='right')


def numbers():
    return json.loads(NUMBERS.read_text())


def gauss():
    return json.loads(GAUSS.read_text())


def sample_name():
    return next(k for k, c in SAMPLES.items() if c['probes'] == PROBES)


def raw_available():
    return all((ARTIFACTS / p / 'plans.npz').exists() for p in PROBES.values())


def plans(model):
    """Plans of one model; from $FROST_ARTIFACTS if present, otherwise the cached example clip(s)."""
    path = ARTIFACTS / PROBES[model] / 'plans.npz'
    if not path.exists():
        path = CACHE / f'plans_{sample_name()}_{model}.npz'
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def compute_tables():
    """Per-clip metrics for every config and model, computed by the paper's own generator from the raw plans."""
    from frost.analysis import paper_numbers as pn
    tags = {k: t for k, t, _ in pn.MODELS}
    pn.MODELS = [(k, tags[k], d) for k, d in PROBES.items()]
    args = argparse.Namespace(data=ARTIFACTS / DATA.name, artifacts=ARTIFACTS, range=RANGE)
    return pn.load(args)


def tables():
    """Per-clip metrics (raw plans if available, else results/figure_cache/per_clip_<sample>.npz)."""
    if raw_available() and (ARTIFACTS / DATA.name).exists():
        return compute_tables()
    with np.load(CACHE / f'per_clip_{sample_name()}.npz') as z:
        names = [str(n) for n in z['names']]
        out = {}
        for key in z.files:
            if key == 'names':
                continue
            model, config = key.split('/')
            arr = z[key]
            out.setdefault(model, {})[config] = {n: {m: float(arr[i, j]) for j, m in enumerate(CACHE_METRICS)} for i, n in enumerate(names)}
    return names, out


def boot_samples(names):
    rng = np.random.default_rng(BOOT_SEED)
    return [list(rng.choice(names, len(names))) for _ in range(BOOT)]


def curve(table, names, metric, samples, scale=1., paired=False):
    """Explicit-temperature curve A_tau: point estimate and 95% clip-bootstrap band.
    paired=True plots the per-clip change relative to A_1, so the band reflects the paired comparison
    rather than between-clip differences in difficulty."""
    configs = [f'heun32_t{t}' for t in TAUS]
    base = np.array([table['heun32_t1.0'][n][metric] for n in names]) if paired else 0.
    arr = {c: np.array([table[c][n][metric] for n in names]) - base for c in configs}
    idx = {n: i for i, n in enumerate(names)}
    point = np.array([arr[c].mean() for c in configs]) * scale
    boots = np.array([[arr[c][[idx[n] for n in s]].mean() for c in configs] for s in samples]) * scale
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    return np.array(TAUS), point, lo, hi


def sampler_ci(table, names, config, metric, samples, scale=1.):
    v = np.array([table[config][n][metric] for n in names]);idx = {n: i for i, n in enumerate(names)}
    b = np.array([v[[idx[n] for n in s]].mean() for s in samples])
    return v.mean() * scale, *(np.quantile(b, [.025, .975]) * scale)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
    print('wrote', path)
