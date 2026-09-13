"""Independent all-model reload, monthly boundary and projection KKT audit."""
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_net_hgb import OUT, BASE, RAW, PRIMARY, AbsoluteNetStore
from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, features_for_day, array_hash, digest, save
from experiments.exp008.forecast_calibration import _daylight
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import split_days


def main():
    data = Data()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    training = json.loads((OUT / 'training_audit.json').read_text())
    old_training = {a['month']: a for a in json.loads((BASE / 'training_audit.json').read_text())}
    raw, full, old_raw = AbsoluteNetStore(RAW), AbsoluteNetStore(), AbsoluteHGBStore('direct_hgb_raw')
    with np.load(OUT / 'raw_absolute_net.npz') as z:
        net = z['net_kw'].copy()
        np.testing.assert_array_equal(z['origins'], raw.origins)
    features = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    rows, projection = [], []
    for record in training:
        month = record['month']
        train, validation, formal, cutoff = split_days(month)
        assert record['training_days'] == train.tolist() and record['validation_days'] == validation.tolist()
        assert record['formal_days'] == formal.tolist()
        assert (train[-1] + 1) * 144 <= record['train_label_stop_exclusive']
        assert (validation[-1] + 1) * 144 == record['validation_label_stop_exclusive'] == formal[0] * 144
        old = old_training[month]
        for new_key, old_key, ids in [('training_feature_sha256', 'training_feature_hash', train),
                                     ('validation_feature_sha256', 'validation_feature_hash', validation)]:
            assert record[new_key] == old[old_key] == array_hash(features[ids - 7].reshape(-1, 31))
        for key, ids in [('training_label_sha256', train), ('validation_label_sha256', validation)]:
            values = data.actual[ids[:, None] * 144 + np.arange(144)]
            expected_net_labels = (values[:, :, 0] - values[:, :, 1]).ravel()
            assert record[key] == array_hash(expected_net_labels)
        v = {r['channel']: r['validation_rmse_kw_before_physical_postprocessing'] ** 2 for r in old['models']}
        assert record['projection']['load_delta_weight'] == v['load'] / (v['load'] + v['pv'])
        path = Path(record['model_path'])
        assert digest(path) == record['model_sha256']
        model = joblib.load(path)
        for key, value in protocol['model_configuration'].items():
            assert model.get_params()[key] == value
        with threadpool_limits(limits=1):
            reloaded = model.predict(features[formal - 7].reshape(-1, 31)).reshape(len(formal), 144)
        np.testing.assert_array_equal(reloaded, net[formal - 31])
        rows.append({'month': month, 'same_train_validation_features_as_original_absolute_HGB': True,
            'absolute_net_labels_independently_rebuilt': True, 'all_formal_predictions_exact_from_reload': True,
            'model_configuration_unchanged': True, 'projection_variance_and_weight_past_validation_only': True})
        for day in formal:
            i = day - 31
            mask = _daylight(data.actual, int(day)).astype(bool)
            expected_net = net[i].copy()
            expected_net[~mask] = np.maximum(expected_net[~mask], 0.)
            # Independent weighted least-squares solution from the normal equation.
            l0, p0 = old_raw.values[i, :, 0], old_raw.values[i, :, 1]
            lstar = (v['pv'] * l0 + v['load'] * (expected_net + p0)) / (v['load'] + v['pv'])
            expected_load = np.maximum(lstar, np.maximum(expected_net, 0.))
            expected_pv = expected_load - expected_net
            expected_load[~mask], expected_pv[~mask] = expected_net[~mask], 0.
            expected = np.column_stack((expected_load, expected_pv))
            np.testing.assert_allclose(raw.values[i], expected, rtol=0, atol=1e-9)
            load, pv = raw.values[i, :, 0], raw.values[i, :, 1]
            gradient = 2 * ((load - l0) / v['load'] + (pv - p0) / v['pv'])
            interior = mask & (load > 1e-6) & (pv > 1e-6)
            boundary = mask & ~interior
            assert np.max(np.abs(gradient[interior]), initial=0.) < 1e-10
            assert np.min(gradient[boundary], initial=0.) >= -1e-10
            assert np.min(raw.values[i]) >= 0 and np.min(full.values[i]) >= 0
            np.testing.assert_array_equal(raw.values[i, ~mask, 1], np.zeros(np.sum(~mask)))
            np.testing.assert_array_equal(full.values[i, ~mask, 1], np.zeros(np.sum(~mask)))
            projection.append({'day': int(day), 'independent_constrained_WLS_solution_matches': True,
                'KKT_interior_and_boundary_conditions_passed': True, 'raw_and_final_nonnegative_night_PV0': True})
    for p, h in protocol['source_sha256'].items():
        assert digest(ROOT / p) == h
    for p, h in protocol['input_artifact_sha256'].items():
        assert digest(p) == h
    result = {'passed': True, 'monthly_model_checks': rows, 'all334_projection_checks': projection,
        'all_sources_and_baseline_artifacts_unchanged': True,
        'no_model_or_projection_weight_selected_by_formal_labels': True,
        'test_source_sha256': digest(__file__)}
    save(OUT / 'independent_verification.json', result)
    shutil.copy2(__file__, OUT / 'independent_audit_source.py')
    print('Passed: 11 complete monthly model reloads/boundaries and 334 independent constrained-projection KKT checks.')


if __name__ == '__main__':
    main()
