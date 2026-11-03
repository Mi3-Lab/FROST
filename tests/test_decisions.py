"""Decision classes (frost.core.decisions): thresholds fixed before any result."""
import numpy as np
import pytest

from frost.core.decisions import DT, HEADING_DEG, OFFSET_M, STOP_EXTENT_M, decisions, plan_speed


def plan(speed=10., lateral=None, n=64):
    """Straight plan at constant speed (x forward, y left), 0.1 s spacing; optional lateral profile y(t)."""
    t = DT * np.arange(1, n + 1)
    y = np.zeros(n) if lateral is None else lateral(t)
    return np.c_[speed * t, y, np.zeros(n)]


def classes(traj, v):
    d = decisions(traj, v)
    return d['longitudinal'], d['lateral']


def test_constant_speed_straight_is_keep_straight():
    assert classes(plan(10.), 10.) == ('keep', 'straight')
    assert abs(plan_speed(plan(10.)) - 10.) < 1e-9


def test_stop_when_extent_within_threshold():
    traj = plan(0.2)  # 64 * 0.1 s * 0.2 m/s = 1.28 m < 1.5 m
    assert np.linalg.norm(traj[:, :2], axis=1).max() <= STOP_EXTENT_M
    assert classes(traj, 5.)[0] == 'stop'


@pytest.mark.parametrize('current,planned,expected', [
    (10., 8.4, 'decelerate'),   # band = max(1, 0.15 * 10) = 1.5 m/s
    (10., 8.8, 'keep'),
    (10., 11.6, 'accelerate'),
    (4., 2.9, 'decelerate'),    # band floor of 1 m/s at low speed
    (4., 3.1, 'keep'),
])
def test_longitudinal_band(current, planned, expected):
    assert classes(plan(planned), current)[0] == expected


def test_lateral_heading_and_offset():
    turn_left = plan(8., lambda t: 0.8 * np.clip(t - 3., 0, None) ** 2)
    assert classes(turn_left, 8.)[1] == 'left'
    turn_right = plan(8., lambda t: -0.8 * np.clip(t - 3., 0, None) ** 2)
    assert classes(turn_right, 8.)[1] == 'right'
    # lane change: final heading ~0, final offset beyond the 1.75 m threshold
    shift = plan(8., lambda t: (OFFSET_M + 0.5) / (1 + np.exp(-4 * (t - 3.))))
    assert classes(shift, 8.)[1] == 'shift_left'
    small = plan(8., lambda t: (OFFSET_M - 0.5) / (1 + np.exp(-4 * (t - 3.))))
    assert classes(small, 8.)[1] == 'straight'


def test_margins_are_zero_at_the_boundary():
    v = 10.;band = max(1., .15 * v)
    d = decisions(plan(v - band), v)
    assert d['long_margin'] < 1e-6
    assert HEADING_DEG == 20. and OFFSET_M == 1.75
