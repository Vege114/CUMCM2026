import numpy as np

from experiments.exp008.load_energy_ar1 import correct_day, ridge_gain


def test_ridge_closed_form_matches_convex_grid_and_limits():
    x = np.array([100., -40., 30.])
    y = np.array([70., -30., 25.])
    gain, fit = ridge_gain(x, y)
    grid = np.linspace(0, 1, 100001)
    loss = ((y[None, :]-grid[:, None]*x)**2/fit['scale_squared_kw2']).sum(1)+7*(grid-.5)**2
    assert abs(gain-grid[loss.argmin()]) < 1e-5
    assert ridge_gain([100.], [10000.])[0] == 1.
    assert ridge_gain([100.], [-10000.])[0] == 0.
    assert ridge_gain([], [])[0] == .5
    assert ridge_gain([0.], [100.])[0] == .5


def test_historical_scale_equivariance_above_floor():
    x, y = np.array([10., -20.]), np.array([15., -25.])
    assert abs(ridge_gain(x, y)[0]-ridge_gain(100*x, 100*y)[0]) < 1e-15


def test_complete_window_and_current_future_prefix_causality():
    origins = np.arange(31, 66)*144
    base = np.full((35, 144, 2), 200.)
    actual = np.full((66*144, 2), 300.)
    expected, audit = correct_day(actual, origins, base, 33)
    assert len(audit['history_origins']) == 28
    assert len(audit['pair_origins']) == 27
    assert audit['maximum_label_index'] == origins[33]-1
    changed = actual.copy()
    changed[origins[33]:] += 50000
    changed_base = base.copy()
    changed_base[34:] += 70000
    measured, _ = correct_day(changed, origins, changed_base, 33)
    np.testing.assert_array_equal(measured, expected)
    np.testing.assert_array_equal(measured[:, 1], base[33, :, 1])
    changed[:origins[33]-28*144] += 90000
    measured, _ = correct_day(changed, origins, changed_base, 33)
    np.testing.assert_array_equal(measured, expected)
