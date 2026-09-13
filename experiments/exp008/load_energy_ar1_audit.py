"""Score the recorded AR1 protocol once, with independent future mutations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_shape_diagnostic import summary
from experiments.exp008.load_energy_ar1 import OUT, AR1MemoryForecasts, AR1MemoryStore, record_protocol
from experiments.exp008.load_energy_memory import EnergyMemoryStore
from experiments.exp008.risk_window import WindowResidualScenarios
from experiments.problem2.exp003.data import Data, ROOT


def run():
    record_protocol()  # Happens before generating or scoring this candidate.
    data = Data()
    candidate = AR1MemoryStore(data=data)
    fixed = EnergyMemoryStore(data=data)
    with np.load(ROOT/'data/results/exp008/load_energy_memory/predictions.npz') as previous:
        fixed_error = float(np.max(np.abs(fixed.values-previous['values'])))
        assert fixed_error == 0
    mutations = []
    for day in (31, 32, 33, 62, 243, 364):
        index = candidate.lookup[day*144]
        actual = copy.copy(data)
        actual.actual = data.actual.copy()
        actual.actual[day*144:] += 50000
        base = copy.copy(candidate.base_store)
        base.values = candidate.base_values.copy()
        base.values[index+1:] += 70000
        altered = AR1MemoryStore(data=actual, base_store=base)
        error = float(np.max(np.abs(candidate.values[:index+1]-altered.values[:index+1])))
        assert error == 0
        risk = WindowResidualScenarios(candidate.origins, candidate.values, data.actual, data.fixed_price)
        risk_alt = WindowResidualScenarios(altered.origins, altered.values, actual.actual, data.fixed_price)
        support_error = float(np.max(np.abs(risk.for_day(day)[0]-risk_alt.for_day(day)[0])))
        assert support_error == 0
        mutations.append({'day': day, 'prediction_prefix_error_kw': error,
                          'tree28_support_error_kwh': support_error})
    adapter = AR1MemoryForecasts()
    adapter_rows = []
    for day in (31, 32, 59, 243, 364):
        p = adapter.get(day)
        np.testing.assert_array_equal(np.column_stack((p['load_kw'], p['pv_kw'])), candidate.get(day*144))
        path = adapter.net_error_paths(day)
        rebuilt = []
        for old in path['origins']//144:
            issue = adapter.get(int(old))
            observed = adapter.data.actual[old*144:(old+1)*144]
            rebuilt.append(((observed[:, 0]-issue['load_kw'])-(observed[:, 1]-issue['pv_kw']))/6)
        error = float(np.max(np.abs(path['errors_kwh']-rebuilt)))
        assert error == 0
        adapter_rows.append({'day': day, 'same_pipeline_risk_error_kwh': error})
    pv_error = float(np.max(np.abs(candidate.values[:, :, 1]-fixed.values[:, :, 1])))
    assert pv_error == 0
    verification = {'passed': True, 'fixed_memory_reconstruction_error_kw': fixed_error,
                    'future_mutations': mutations, 'adapter_checks': adapter_rows,
                    'pv_max_difference_kw': pv_error,
                    'all_gains_bounded': all(0 <= a['gain'] <= 1 for a in candidate.audit),
                    'all_labels_completed': all(a['maximum_label_index'] is None or
                        a['maximum_label_index'] < a['origin'] for a in candidate.audit),
                    'max_pairs': max(len(a['pair_origins']) for a in candidate.audit)}
    np.savez_compressed(OUT/'predictions.npz', origins=candidate.origins,
                        values=candidate.values, base_values=candidate.base_values,
                        delta=candidate.delta, gains=np.array([a['gain'] for a in candidate.audit]))
    (OUT/'prediction_audit.json').write_text(json.dumps(candidate.audit, indent=2))
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2))
    # Actual future labels enter only the following retrospective score stage.
    truth = data.actual[candidate.origins[:, None]+np.arange(144)]
    dates = pd.date_range('2025-02-01', '2025-12-31')
    groups = {'all334': np.ones(334, bool), 'months_6_7_9': dates.month.isin([6, 7, 9]),
              'month_first7': dates.day <= 7, 'month_rest': dates.day > 7,
              **{f'month_{m}': dates.month == m for m in range(2, 13)}}
    rows = []
    for name, values in [('fixed_half', fixed.values), ('ar1_w28_prior7', candidate.values)]:
        error = values-truth
        for channel, e in [('load', error[:, :, 0]), ('pv', error[:, :, 1]),
                           ('net', error[:, :, 0]-error[:, :, 1])]:
            for group, ids in groups.items():
                rows.append({'name': name, 'channel': channel, 'group': group,
                             'days': int(np.sum(ids)), **summary(e[ids], data.fixed_price)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'forecast_comparison.csv', index=False)
    coefficients = pd.DataFrame({'day': candidate.origins//144, 'date': dates.astype(str),
        'gain': [a['gain'] for a in candidate.audit],
        'history_days': [len(a['history_origins']) for a in candidate.audit],
        'pairs': [len(a['pair_origins']) for a in candidate.audit],
        'load_correction_kw': [a['load_correction_kw'] for a in candidate.audit]})
    coefficients.to_csv(OUT/'daily_coefficients.csv', index=False)
    check = table[(table.channel == 'net') & (table.group == 'all334')].set_index('name')
    metrics = ['daily_energy_rmse_kwh', 'cumulative_error_rmse_kwh', 'high_price_rmse_kw']
    comparisons = {metric: {'fixed_half': float(check.loc['fixed_half', metric]),
                           'ar1': float(check.loc['ar1_w28_prior7', metric]),
                           'improved': bool(check.loc['ar1_w28_prior7', metric] <
                                            check.loc['fixed_half', metric])} for metric in metrics}
    gate = {'passed': all(r['improved'] for r in comparisons.values()), 'comparisons': comparisons,
            'action_if_failed': 'stop_without_LP_bridge_or_parameter_search',
            'coefficient_distribution': coefficients.gain.describe().to_dict(),
            'gain_at_zero_days': int(np.sum(coefficients.gain == 0)),
            'gain_at_one_days': int(np.sum(coefficients.gain == 1)),
            'audit_passed': verification['passed']}
    (OUT/'prediction_gate.json').write_text(json.dumps(gate, indent=2))
    manifest = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'protocol_sha256': hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                'source_data_hashes': data.hashes,
                'underlying_values_sha256': hashlib.sha256(candidate.base_values.tobytes()).hexdigest(),
                'fixed_memory_values_sha256': hashlib.sha256(fixed.values.tobytes()).hexdigest(),
                'archive_sha256': hashlib.sha256((OUT/'predictions.npz').read_bytes()).hexdigest(),
                'complete': True, 'days': 334, 'prediction_gate': gate['passed'],
                'LP_bridge_performed': False}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT/'audit_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(gate, indent=2), flush=True)
    return gate


if __name__ == '__main__':
    run()
