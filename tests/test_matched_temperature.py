"""Effective-temperature matching on synthetic curves (frost.analysis.paper_numbers, frost.core.matched_temperature_stats)."""
import numpy as np

from frost.analysis import paper_numbers as pn
from frost.core import matched_temperature_stats as mts

NAMES = [f'context_{i:03d}' for i in range(5)]


def table(spread_of, metric_of, extra):
    """Per-clip table where every clip of config c has spread spread_of(tau) and metric metric_of(tau)."""
    t = {}
    for tau in pn.TAUS:
        t[f'heun32_t{tau}'] = {n: {'spread': spread_of(tau), 'ade1': metric_of(tau), 'minade6': 0., 'flip': 0., 'entropy': 0.} for n in NAMES}
    t.update(extra)
    return t


def entry(spread, ade):
    return {n: {'spread': spread, 'ade1': ade, 'minade6': 0., 'flip': 0., 'entropy': 0.} for n in NAMES}


def test_tau_eff_recovers_the_temperature_of_a_sampler_on_the_curve():
    spread_of = lambda tau: 2. * tau
    metric_of = lambda tau: 1. + 0.5 * tau ** 2
    for tau in (0.45, 0.62, 0.84, 1.0):
        t = table(spread_of, metric_of, {'euler4_t1.0': entry(spread_of(tau), metric_of(tau))})
        r = pn.matched(t, NAMES, 'euler4_t1.0')
        # linear interpolation of a quadratic metric between grid points: exact tau, metric within the chord error
        assert abs(r['tau_eff'] - tau) < 1e-12 and r['in_range']
        assert abs(r['ade1']['residual']) < 0.5 * 0.05 ** 2 + 1e-12
        assert abs(r['ade1']['effect'] - (metric_of(tau) - metric_of(1.))) < 1e-12


def test_residual_is_the_vertical_distance_to_the_curve():
    t = table(lambda tau: 2. * tau, lambda tau: tau, {'euler4_t1.0': entry(1.4, 0.7 + 0.03)})
    r = pn.matched(t, NAMES, 'euler4_t1.0')
    assert abs(r['tau_eff'] - 0.7) < 1e-12 and abs(r['ade1']['residual'] - 0.03) < 1e-12


def test_out_of_range_spread_is_flagged():
    t = table(lambda tau: 2. * tau, lambda tau: tau, {'heun5_t1.0': entry(2.5, 1.)})
    r = pn.matched(t, NAMES, 'heun5_t1.0')
    assert not r['in_range'] and r['tau_eff'] == 1.0  # np.interp clamps; the flag carries the information


def test_matched_temperature_stats_summarize_agrees():
    curve = [f'heun32_t{tau}' for tau in pn.TAUS]
    t = {c: {n: {m: 0. for m in mts.METRICS} | {'spread': 2. * tau, 'ade1': tau} for n in NAMES} for c, tau in zip(curve, pn.TAUS)}
    t['euler10_t1.0'] = {n: {m: 0. for m in mts.METRICS} | {'spread': 1.7, 'ade1': 0.85} for n in NAMES}
    r = mts.summarize(t, NAMES, curve, pn.TAUS, 'euler10_t1.0')
    assert abs(r['tau_eff'] - 0.85) < 1e-12 and abs(r['ade1']['residual']) < 1e-12
