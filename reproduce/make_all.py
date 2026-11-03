"""Regenerate every table, macro file and figure of the paper from results/ (CPU, a few minutes).

    python reproduce/make_all.py                 # tables and figures
    python reproduce/make_all.py --tables-only   # LaTeX tables and macros only (no matplotlib needed)
    python reproduce/make_all.py --check         # also compare the tables with the published SHA-256 digests

Outputs go to $FROST_OUTPUT (default reproduce/output/{tables,figures}). File names are mapped to the names used by
the paper sources in repro_paths.PAPER_NAMES.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from repro_paths import FIGURES, PAPER_NAMES, TABLES  # noqa: E402

TABLE_SCRIPTS = ['make_tables.py', 'make_appendix_numbers.py', 'make_crossmodel_numbers.py', 'make_robustness.py',
                 'make_closedloop_table.py', 'make_smolvla_numbers.py']
FIGURE_SCRIPTS = [('make_fig_teaser_wide.py', []), ('make_fig_test.py', []), ('make_fig_mechanism.py', []),
                  ('make_fig_curves.py', ['--rows', 'ade1,flip', '--out', 'fig_curves_main.pdf']),
                  ('make_fig_frost_atlas.py', []), ('make_fig_frost_summary.py', []), ('make_fig_wod.py', [])]
GOLDEN = HERE.parent / 'tests' / 'golden' / 'tables_sha256.json'


def run(script, args=()):
    env = {**os.environ, 'MPLBACKEND': os.environ.get('MPLBACKEND', 'Agg'), 'PYTHONDONTWRITEBYTECODE': '1'}
    subprocess.run([sys.executable, str(script), *args], check=True, env=env, cwd=script.parent, stdout=subprocess.DEVNULL)
    print('ok', script.relative_to(HERE.parent), *args)


def check():
    """Compare every generated table with the SHA-256 of the file the paper was built from."""
    golden = json.loads(GOLDEN.read_text());bad = []
    for name, digest in sorted(golden.items()):
        path = TABLES / name
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if got != digest:
            bad.append(name)
    print(f'{len(golden) - len(bad)}/{len(golden)} tables byte-identical to the paper' + (f'; differ: {bad}' if bad else ''))
    return not bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables-only', action='store_true');p.add_argument('--check', action='store_true');a = p.parse_args()
    for s in TABLE_SCRIPTS:
        run(HERE / 'tables' / s)
    if not a.tables_only:
        for s, args in FIGURE_SCRIPTS:
            run(HERE / 'figures' / s, args)
    print('tables ->', TABLES);print('figures ->', FIGURES) if not a.tables_only else None
    if a.check and not check():
        sys.exit(1)


if __name__ == '__main__':
    main()
