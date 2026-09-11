"""Perturb all post-January information and compare formal calibration inputs."""

import copy
import json

import numpy as np

from .data import ROOT, Data, split_origins
from .predict import ForecastStore
from .scenarios import ScenarioFactory


def run(run_id="exp002"):
    data = Data()
    changed = copy.deepcopy(data)
    cutoff = 31*144
    changed.actual[cutoff:] *= 100
    changed._forecasts = {o: values*(100 if o >= cutoff else 1)
                          for o, values in data.forecasts.items()}
    training, validation, train_cutoff = split_origins(2)
    np.testing.assert_array_equal(data.labels(np.r_[training, validation]),
                                  changed.labels(np.r_[training, validation]))
    for a, b in zip(data.features(validation, train_cutoff),
                    changed.features(validation, train_cutoff)):
        np.testing.assert_array_equal(a, b)
    a_store = ForecastStore(data, run_id, 42)
    b_store = ForecastStore(changed, run_id, 42)
    a_factory, b_factory = ScenarioFactory(data, a_store), ScenarioFactory(changed, b_store)
    checked = 0
    for scenario in ("2", "3", "4-2", "4-3"):
        issued = scenario in ("3", "4-3")
        for day in range(24, 31):
            for slot in (0, 36, 72, 108) if issued else (0,):
                origin = day*144+slot
                a = a_store.get(origin, issued=issued, validation_month=2)
                b = b_store.get(origin, issued=issued, validation_month=2)
                np.testing.assert_array_equal(a, b)
                left = a_factory.build(origin, a, scenario)
                right = b_factory.build(origin, b, scenario)
                for name in ("paths", "groups", "probabilities"):
                    np.testing.assert_array_equal(left[name], right[name])
                assert left["metadata"] == right["metadata"]
                np.testing.assert_array_equal(data.actual[origin:(day+1)*144],
                                              changed.actual[origin:(day+1)*144])
                checked += 1
    result = {"status": "passed", "perturbation": "All February-and-later actuals and forecast releases multiplied by 100",
              "formal_checkpoint": "m02_mlp_42", "training_labels_and_validation_features": "identical",
              "calibration_origins_across_four_questions": checked,
              "predictions_scenario_paths_probabilities_information_tree": "bitwise identical",
              "january_replay_actuals": "identical", "additional_fits": 0,
              "scope": "Calibration optimizer inputs and realized January settlement inputs are invariant. Tiny fully solved purchase invariance is checked separately in unit tests; wall-clock bounded searches can vary between identical reruns."}
    (ROOT / "data/results" / run_id / "calibration_causality_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    run()
