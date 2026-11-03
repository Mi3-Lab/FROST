"""Every module compiles; everything except the model probes imports without torch."""
import importlib
import importlib.util
import py_compile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = sorted('.'.join(p.relative_to(REPO).with_suffix('').parts) for p in (REPO / 'frost').rglob('*.py') if p.name != '__init__.py')
NEEDS = {'frost.analysis.closedloop_analysis': 'polars', 'frost.analysis.closedloop_comfort': 'alpasim_utils',
         'frost.analysis.navsim_analysis': 'pandas', 'frost.data.physicalai_contexts': 'pandas', 'frost.data.split_overlap': 'pandas',
         'frost.data.simwam_ego_inputs': 'torch'}


@pytest.mark.parametrize('path', sorted(p for d in ('frost', 'reproduce', 'scripts', 'deploy') for p in (REPO / d).rglob('*.py')))
def test_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / 'x.pyc'), doraise=True)


@pytest.mark.parametrize('name', [m for m in MODULES if '.probes.' not in m])
def test_imports_without_model_stack(name):
    dep = NEEDS.get(name)
    if dep and importlib.util.find_spec(dep) is None:
        pytest.skip(f'{dep} not installed')
    importlib.import_module(name)
