"""Step-grid construction of the schedule study (frost.core.schedules). Ported from the research test suite."""
import numpy as np

from frost.core.schedules import grid_from_density


def test_grid_from_uniform_density_is_uniform():
    times = np.linspace(0, 1, 200, endpoint=False)
    np.testing.assert_allclose(grid_from_density(times, np.ones(200), 4), [0, .25, .5, .75, 1], atol=1e-9)


def test_grid_concentrates_where_density_is_high():
    times = np.linspace(0, 1, 200, endpoint=False);density = np.where(times < .5, 9., 1.)
    grid = grid_from_density(times, density, 4)
    assert np.all(grid[1:4] < .5) and grid[0] == 0 and grid[-1] == 1
