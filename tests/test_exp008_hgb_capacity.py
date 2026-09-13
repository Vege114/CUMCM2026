from types import SimpleNamespace

import numpy as np

from experiments.exp008.hgb_capacity_selection import choose
from experiments.exp008.hgb_stage_selection import best_stage


def test_capacity_score_tie_prefers_smaller_family():
    rows = [{'leaf_nodes': leaf, 'best_validation_score': score}
            for leaf, score in ((63, -7.), (31, -7.), (15, -7.))]
    assert choose(rows)['leaf_nodes'] == 15
    rows[-1]['best_validation_score'] = -8.
    assert choose(rows)['leaf_nodes'] == 31


def test_nonzero_stage_selection_ignores_intercept_and_prefers_earliest_tie():
    model = SimpleNamespace(validation_score_=np.array([100., -5., -3., -3., -8.]))
    assert best_stage(model) == 2
    assert best_stage(SimpleNamespace(validation_score_=np.array([0., -20.]))) == 1
