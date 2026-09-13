import itertools

import numpy as np

from experiments.exp008.mode_minhold_physical import mode_state, plan


def test_hold_windows_equal_independent_completed_run_rule():
    for old, age in itertools.product((-1, 1), (1, 2, 3, 8)):
        for bits in itertools.product((False, True), repeat=6):
            bits = np.array(bits)
            flips = np.r_[bits[0] != (old > 0), bits[1:] != bits[:-1]]
            permitted = (all(np.sum(flips[t:t+3]) <= 1 for t in range(6)) and
                         all(bits[t] == (old > 0) for t in range(max(0, 3-age))))
            try:
                mode_state(bits, old, age)
                checked = True
            except AssertionError:
                checked = False
            assert permitted == checked


def test_cross_day_hold_obligation_is_not_reset():
    state, _ = mode_state([True, True, False, False, False, True], 1, 1)
    assert state['final_planned_mode'] == 1
    assert state['final_planned_run_slots'] == 1
    assert state['next_day_required_hold_slots'] == 2
    paths = np.array([[100., 200., 300., 400., 500., 600.],
                      [200., 100., 500., 100., 600., 400.],
                      [-100., 300., -200., 500., 200., 500.]])
    result = plan(paths, np.array([1.5, 1.5, .2, .2, .5, 1.5]), 6000.,
                  state['final_planned_mode'], state['final_planned_run_slots'], final=True)
    assert result['allowed_charge'][:2].all()
    assert result['metadata']['scenario_physical_checks']['passed']
    assert result['metadata']['scenario_physical_checks']['mode_flip_XOR_error'] < 1e-6
