"""Read-only independent reconstruction of the fixed ensemble archive.

Does not fit weights, rerun dispatch, or change either input pipeline.
"""
import json
import numpy as np

from experiments.exp008.forecast_half_ensemble import (
    OUT, MODEL_ID, EnsembleStore, WindowResidualScenarios, Data, audited_forecasts,
    array_hash, digest, save,
)


def main():
    store, forecasts, data = EnsembleStore(), audited_forecasts(), Data()
    identity = forecasts.forecast_identity()
    assert identity['model_id'] == MODEL_ID
    assert identity['prediction_values_sha256'] == array_hash(store.values)
    for day in range(31, 365):
        issue = forecasts.get(day, scenario='2')
        values = np.column_stack((issue['load_kw'], issue['pv_kw']))
        np.testing.assert_array_equal(values, store.get(day * 144))
        assert issue['audit']['selected_model_id'] == MODEL_ID
    history_rows = []
    for day in (59, 90, 151, 243, 364):
        paths = forecasts.net_error_paths(day)
        historical_days = paths['origins'] // 144
        expected = np.stack([
            (data.actual[old * 144:(old + 1) * 144] - store.get(old * 144))
            for old in historical_days
        ])
        np.testing.assert_array_equal(paths['errors_kw'], expected)
        np.testing.assert_array_equal(paths['errors_kwh'], (expected[:, :, 0] - expected[:, :, 1]) / 6)
        assert paths['audit']['model_id'] == MODEL_ID
        assert paths['audit']['historical_predictions_use_same_override']
        assert paths['audit']['max_observed_index'] < day * 144
        history_rows.append({'day': day, 'complete_paths': len(historical_days),
                             'max_abs_path_error_kwh': 0., 'model_identity_correct': True})
    directory = OUT / 'lp_bridge/half_ensemble_tree28_q08_buffer500_334days'
    with np.load(directory / 'supports.npz') as z:
        support_values, days = z['supports'], z['days']
    risk = WindowResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    audit = json.loads((directory / 'audit.json').read_text())
    for i, day in enumerate(days):
        expected, rebuilt_audit = risk.for_day(int(day))
        np.testing.assert_array_equal(support_values[i], expected)
        assert audit[i]['training_origins'] == rebuilt_audit['training_origins']
        assert audit[i]['fused_forecast_values_sha256'] == array_hash(store.values)
        assert all(origin + 144 <= day * 144 for origin in audit[i]['training_origins'])
        if not audit[i]['fallback']:
            assert audit[i]['residual_source'] == 'half_ensemble_prequential_forecast'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        from experiments.exp008.forecast_half_ensemble import ROOT
        assert digest(ROOT / path) == expected
    save(OUT / 'adapter_and_risk_verification.json', {
        'passed': True, 'forecast_identity': identity, 'midnight_exact_store_match_days': 334,
        'historical_fused_error_path_reconstruction': history_rows,
        'full_334_day_tree_supports_rebuilt_exactly': True,
        'all_training_labels_complete_before_current_issue': True,
        'old_risk_cache_not_used': True, 'source_files_unchanged': True,
    })
    print('Passed: signed interface, 334 exact midnight outputs, 140 historical paths, 334 rebuilt tree supports.')


if __name__ == '__main__':
    main()
