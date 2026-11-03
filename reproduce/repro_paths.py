"""Input and output locations of the table and figure generators.

FROST_RESULTS   released summaries (default: <repo>/results); every table and figure reads only from here
FROST_ARTIFACTS full artifact tree with per-clip plans (only needed to rebuild results/figure_cache or the
                closed-loop exploratory cache; default: <repo>/artifacts)
FROST_OUTPUT    where the generated .tex and .pdf files are written (default: <repo>/reproduce/output)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get('FROST_RESULTS', REPO / 'results'))
ARTIFACTS = Path(os.environ.get('FROST_ARTIFACTS', REPO / 'artifacts'))
OUTPUT = Path(os.environ.get('FROST_OUTPUT', REPO / 'reproduce' / 'output'))
TABLES = OUTPUT / 'tables'
FIGURES = OUTPUT / 'figures'
for _d in (TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Release file name -> file name used by the paper sources (\input{tables/<name>}). Files not listed keep their name.
PAPER_NAMES = {
    'numbers_controls.tex': 'numbers_v533.tex',
    'numbers_reference_floor.tex': 'numbers_v542.tex',
    'numbers_wod.tex': 'numbers_v543.tex',
    'numbers_baselines.tex': 'numbers_v61.tex',
    'numbers_wod_secondary.tex': 'numbers_v62.tex',
    'numbers_wod_official.tex': 'numbers_v63.tex',
    'numbers_crossmodel.tex': 'numbers_v7.tex',
    'tab_navsim_merged.tex': 'tab_navsim_v60.tex',
    'tab_onetwo.tex': 'tab_onetwo_v60.tex',
}
