"""Clip bootstrap and TOST equivalence of frost.analysis.paper_numbers on a synthetic probe with known answers.

Plans are mean + c * z with shared per-seed noise z, so a sampler that is exactly 'cooler' (c on the tau grid) lies on
the reference curve A_tau = mean + tau * z: its tau_eff is c and its residuals vanish in every bootstrap resample. A sampler at c = 1 whose plans
change lateral class for half of the seeds has the reference spread but extra decision flips: not equivalent.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = list(range(9201, 9207))
SCALE = {'euler1': 0., 'euler2': .3, 'euler4': .6, 'euler6': .7, 'euler10': .8, 'heun3': 1., 'heun5': 1.}
TAUS = [1., .9, .8, .7, .6, .5, .4]


def build(root, n=24, rng=np.random.default_rng(0)):
    data, probe = root / 'contexts', root / 'probe'
    data.mkdir();probe.mkdir()
    t = 0.1 * np.arange(1, 65);records, plans = [], {}
    for i in range(n):
        name = f'context_{i:03d}';v = rng.uniform(6, 12)
        mean = np.c_[v * t, 0.05 * rng.standard_normal() * t, np.zeros(64)]
        np.savez(data / f'{name}.npz', ego_future_xyz=mean + [0, 0.3, 0])
        records.append({'name': name, 'speed_mps': v})
        for s in SEEDS:
            z = np.cumsum([0.1, 0.25, 0.] * rng.standard_normal((64, 3)), 0)  # lateral noise large enough to change classes
            for tau in TAUS:
                plans[f'{name}_s{s}_heun32_t{tau}'] = (mean + tau * z).astype(np.float32)
            for cfg, c in SCALE.items():
                p = mean + c * z
                if cfg == 'heun5' and s % 2:
                    p = p + np.c_[np.zeros(64), 3.0 * np.clip(t - 3, 0, None) / 3.4, np.zeros(64)]  # ends > 1.75 m left
                plans[f'{name}_s{s}_{cfg}_t1.0'] = p.astype(np.float32)
    (data / 'records.json').write_text(json.dumps(records))
    np.savez(probe / 'plans.npz', **plans)
    (probe / 'manifest.json').write_text(json.dumps({'seeds': SEEDS}))
    return data, probe, n


def test_bootstrap_tost_on_synthetic_probe(tmp_path):
    data, probe, n = build(tmp_path)
    out = tmp_path / 'out'
    cmd = [sys.executable, '-m', 'frost.analysis.paper_numbers', '--data', str(data), '--artifacts', str(tmp_path),
           '--out', str(out), '--macros', str(tmp_path / 'm.tex'), '--range', '0', str(n), '--boot', '300',
           '--probes', 'alpamayo15=probe']
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True)
    m = json.loads((out / 'summary.json').read_text())['models']['alpamayo15']
    for k, c in (('euler4', .6), ('euler10', .8)):
        e = m[f'{k}_t1.0']
        assert abs(e['tau_eff'][0] - c) < 1e-6, (k, e['tau_eff'])
        assert e['tau_eff'][1] <= e['tau_eff'][0] <= e['tau_eff'][2]
        assert e['ade1']['equivalent'] and e['flip']['equivalent'], k
        assert abs(e['flip']['residual'][0]) < 1e-9 and e['flip']['effect'][0] > 0.05  # real effect, fully explained by temperature
    h = m['heun5_t1.0']
    assert h['flip']['sampler'] > 0.3 and not h['flip']['equivalent']
    assert h['flip']['residual'][1] > 0.02  # the whole 95% interval is above the margin
    # The macros file is written with the same numbers.
    assert '\\newcommand{\\NClips}{%d}' % n in (tmp_path / 'm.tex').read_text()
