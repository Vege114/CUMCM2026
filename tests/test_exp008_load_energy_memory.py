import numpy as np

from experiments.exp008.load_energy_memory import correct_day


def arrays():
    origins = np.arange(31, 35)*144
    base = np.full((4, 144, 2), 200.)
    base[:, :, 1] = np.arange(144)[None, :]
    actual = np.full((35*144, 2), 200.)
    actual[31*144:32*144, 0] = 320.
    return actual, origins, base


def test_load_energy_formula_uses_underlying_prior_output_and_preserves_pv():
    actual, origins, base = arrays()
    first, _ = correct_day(actual, origins, base, 0)
    np.testing.assert_array_equal(first, base[0])
    output, audit = correct_day(actual, origins, base, 1)
    np.testing.assert_array_equal(output[:, 0], np.full(144, 260.))
    np.testing.assert_array_equal(output[:, 1], base[1, :, 1])
    assert (output[:, 0]-base[1, :, 0]).sum()/6 == .5*(120*24)
    assert audit['prior_label_end_exclusive'] == origins[1]
    assert audit['recursive_corrected_output_residual'] is False


def test_current_future_actual_and_future_forecasts_do_not_change_output():
    actual, origins, base = arrays()
    expected, _ = correct_day(actual, origins, base, 1)
    changed = actual.copy()
    changed[origins[1]:] += 1e7
    future = base.copy()
    future[2:] += 2e7
    output, _ = correct_day(changed, origins, future, 1)
    np.testing.assert_array_equal(output, expected)


def test_only_preceding_complete_day_is_used_and_negative_load_is_clipped():
    actual, origins, base = arrays()
    base[0, :, 0] = 2000.
    expected, _ = correct_day(actual, origins, base, 1)
    np.testing.assert_array_equal(expected[:, 0], np.zeros(144))
    changed = actual.copy()
    changed[:origins[0]] += 3e7
    output, _ = correct_day(changed, origins, base, 1)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(output[:, 1], base[1, :, 1])
