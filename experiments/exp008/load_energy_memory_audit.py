"""Independent prefix perturbations and fixed-candidate forecast scoring."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from experiments.exp008.forecast_shape_diagnostic import summary
from experiments.exp008.load_energy_memory import (
    OUT, EnergyMemoryForecasts, EnergyMemoryStore, correct_day, save,
)
from experiments.exp008.risk_window import WindowResidualScenarios
from experiments.problem2.exp003.data import Data


def run():
    data = Data()
    memory = save()
    original = memory.base_values
    dates = pd.date_range('2025-02-01', '2025-12-31')
    mutation = []
    for day in (31, 32, 33, 62, 243, 364):
        index = memory.lookup[day*144]
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day*144:] += 50000
        future_base = copy.copy(memory.base_store)
        future_base.values = original.copy()
        future_base.values[index+1:] += 70000
        altered = EnergyMemoryStore(data=changed, base_store=future_base)
        prefix_error = float(np.max(np.abs(altered.values[:index+1]-memory.values[:index+1])))
        assert prefix_error == 0
        risk = WindowResidualScenarios(memory.origins, memory.values, data.actual, data.fixed_price)
        changed_risk = WindowResidualScenarios(altered.origins, altered.values, changed.actual, data.fixed_price)
        support, audit = risk.for_day(day)
        support_error = float(np.max(np.abs(support-changed_risk.for_day(day)[0])))
        assert support_error == 0
        assert audit['max_observed_index'] < day*144
        correction = memory.audit[index]['load_correction_kw']
        manual, _ = correct_day(data.actual, memory.origins, original, index)
        np.testing.assert_array_equal(manual, memory.values[index])
        mutation.append({'day': day, 'prefix_forecast_error_kw': prefix_error,
                         'tree_support_error_kwh': support_error,
                         'load_correction_kw': correction,
                         'risk_fallback': audit['fallback']})
    adapter = EnergyMemoryForecasts()
    adapter_checks = []
    for day in (31, 32, 59, 243, 364):
        predicted = adapter.get(day)
        np.testing.assert_array_equal(
            np.column_stack((predicted['load_kw'], predicted['pv_kw'])),
            memory.get(day*144))
        risk = adapter.net_error_paths(day)
        expected = []
        for old in risk['origins']//144:
            p = adapter.get(int(old))
            observed = adapter.data.actual[old*144:(old+1)*144, :2]
            expected.append(((observed[:, 0]-p['load_kw'])-(observed[:, 1]-p['pv_kw']))/6)
        error = float(np.max(np.abs(risk['errors_kwh']-expected)))
        assert error == 0
        adapter_checks.append({'day': day, 'same_pipeline_error_kwh': error,
                               'periodic_fallback_days': risk['audit']['fallback_days']})
    pv_error = float(np.max(np.abs(memory.values[:, :, 1]-original[:, :, 1])))
    assert pv_error == 0
    truth = data.actual[memory.origins[:, None]+np.arange(144)]
    groups = {'all334': np.ones(334, dtype=bool),
              'months_6_7_9': dates.month.isin([6, 7, 9]),
              'other_months': ~dates.month.isin([6, 7, 9]),
              'month_first7': dates.day <= 7, 'month_rest': dates.day > 7}
    rows = []
    for name, values in [('joint_ridge28', original), ('memory_half', memory.values)]:
        error = values-truth
        for channel, e in [('load', error[:, :, 0]), ('pv', error[:, :, 1]),
                           ('net', error[:, :, 0]-error[:, :, 1])]:
            for group, ids in groups.items():
                rows.append({'name': name, 'channel': channel, 'group': group,
                             'days': int(np.sum(ids)), **summary(e[ids], data.fixed_price)})
            for month in range(2, 13):
                ids = dates.month == month
                rows.append({'name': name, 'channel': channel, 'group': f'month_{month}',
                             'days': int(np.sum(ids)), **summary(e[ids], data.fixed_price)})
    pd.DataFrame(rows).to_csv(OUT/'forecast_comparison.csv', index=False)
    verification = {'passed': True, 'future_actual_and_future_base_prefix_checks': mutation,
                    'adapter_and_historical_risk_checks': adapter_checks,
                    'pv_max_difference_kw': pv_error,
                    'first_forecast_unchanged': bool(np.array_equal(memory.values[0], original[0])),
                    'all_labels_strictly_prior_complete_days': all(
                        row['prior_label_end_exclusive'] is None or
                        row['prior_label_end_exclusive'] <= row['origin'] for row in memory.audit),
                    'coef_fixed': .5, 'no_forecast_model_training': True}
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2))
    print(pd.DataFrame(rows).query('channel == "net" and not group.str.startswith("month_")',
                                 engine='python').to_string(index=False), flush=True)
    return verification


if __name__ == '__main__':
    run()
