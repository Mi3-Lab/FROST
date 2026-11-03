"""End-to-end: regenerating the tables and macro files from results/ reproduces, byte for byte, the files the paper
was built from (SHA-256 digests in tests/golden/tables_sha256.json); the figure scripts run from results/."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((REPO / 'tests/golden/tables_sha256.json').read_text())


def run_all(out, *extra):
    env = {**os.environ, 'FROST_OUTPUT': str(out), 'FROST_RESULTS': str(REPO / 'results'),
           'FROST_ARTIFACTS': str(out / 'no_raw_artifacts'), 'MPLBACKEND': 'Agg'}
    subprocess.run([sys.executable, str(REPO / 'reproduce/make_all.py'), *extra], check=True, env=env, capture_output=True)


@pytest.fixture(scope='module')
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp('repro')
    run_all(out, '--tables-only')
    return out / 'tables'


def test_every_paper_table_is_generated(tables):
    assert sorted(p.name for p in tables.glob('*.tex')) == sorted(GOLDEN)


@pytest.mark.parametrize('name', sorted(GOLDEN))
def test_table_is_byte_identical(tables, name):
    assert hashlib.sha256((tables / name).read_bytes()).hexdigest() == GOLDEN[name]


@pytest.mark.skipif(importlib.util.find_spec('matplotlib') is None, reason='figures extra not installed')
def test_figures_run_from_results(tmp_path):
    run_all(tmp_path)
    figs = {p.name for p in (tmp_path / 'figures').glob('*.pdf')}
    assert {'fig_teaser_wide.pdf', 'fig_test.pdf', 'fig_mechanism.pdf', 'fig_curves_main.pdf', 'fig_frost_atlas.pdf',
            'fig_frost_summary.pdf', 'fig_wod.pdf'} <= figs
