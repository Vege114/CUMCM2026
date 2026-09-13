"""Single causal daily-mean load calibration of the fixed absolute HGB pipeline.

This is a development experiment. Coefficients use only earlier issued errors;
no today's actual, future forecast, or scenario billing enters the correction.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, array_hash
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.problem2.exp003.data import Data

OUT = Path('data/results/exp008/load_mean_state_calibration')
CONFIG = dict(window=56, half_life=28, minimum_history=14,
              ridge=7., intercept_ridge=2., minimum_feature_scale_kw=100.)
NAMES = ['intercept', 'lag1_issued_load_mean_error', 'lag7_issued_load_mean_error',
         'last7_issued_load_mean_error', 'current_minus_yesterday_forecast_mean',
         'weekend_change']


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def feature(actual, values, origins, index):
    day = int(origins[index] // 144)
    ids = np.arange(max(0, index - 7), index)
    if len(ids):
        observed = actual[origins[ids, None]+np.arange(144), 0].mean(axis=1)
        errors = observed - values[ids, :, 0].mean(axis=1)
        lag1, lag7, mean7 = float(errors[-1]), float(errors[0]), float(errors.mean())
        shift = float(values[index, :, 0].mean() - values[index-1, :, 0].mean())
    else:
        lag1 = lag7 = mean7 = shift = 0.
    weekday = (day+2) % 7
    weekend_change = float(weekday >= 5) - float((weekday-1) % 7 >= 5)
    return np.array([1., lag1, lag7, mean7, shift, weekend_change])


def correct(actual, values, origins, index):
    ids = np.arange(max(0, index-CONFIG['window']), index)
    audit = {'origin': int(origins[index]), 'training_origins': origins[ids].tolist(),
             'source_model': 'direct_hgb_ridge28_memory_half', 'current_truth_used': False}
    result = values[index].copy()
    if len(ids) < CONFIG['minimum_history']:
        audit.update(cold_start=True, correction_kw=0.)
        return result, audit
    x = np.stack([feature(actual, values, origins, int(i)) for i in ids])
    xt = feature(actual, values, origins, index)
    observed = actual[origins[ids, None]+np.arange(144), 0].mean(axis=1)
    target = (observed-values[ids, :, 0].mean(axis=1))/100.
    scale = np.maximum(np.std(x, axis=0), CONFIG['minimum_feature_scale_kw'])
    scale[[0, -1]] = 1.
    x, xt = x/scale, xt/scale
    weights = 2**(-np.arange(len(ids)-1, -1, -1)/CONFIG['half_life'])
    penalty = np.full(x.shape[1], CONFIG['ridge'])
    penalty[0] = CONFIG['intercept_ridge']
    coefficients = np.linalg.solve((x*weights[:,None]).T@x+np.diag(penalty),
                                    (x*weights[:,None]).T@target)
    raw = 100.*float(xt@coefficients)
    bound = max(100., 3*float(np.sqrt(np.mean((100*target)**2))))
    correction = float(np.clip(raw, -bound, bound))
    result[:, 0] = np.maximum(0., result[:, 0]+correction)
    audit.update(cold_start=False, correction_kw=correction, coefficients=coefficients.tolist(),
                 feature_scale=scale.tolist(), maximum_label_exclusive=int(origins[ids[-1]]+144),
                 target_rule='observed_minus_unmodified_underlying_issued_load_mean')
    return result, audit


def main():
    if OUT.exists():
        raise FileExistsError('Completed experiments are immutable')
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json', {'one_fixed_candidate': True, 'config': CONFIG, 'features': NAMES,
        'prediction': 'one load-only constant daily shift; PV unchanged; no recursive correction',
        'continuation_gate': 'improve net RMSE, high-price net RMSE, daily-energy and prefix-energy RMSE versus absolute HGB; otherwise no LP bridge',
        'development_year_not_independent_test': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data, base = Data(), AbsoluteHGBStore()
    outputs = [correct(data.actual, base.values, base.origins, i) for i in range(len(base.origins))]
    values = np.stack([r[0] for r in outputs]); audits = [r[1] for r in outputs]
    save(OUT/'causal_fit_audit.json', audits)
    truth = data.actual[base.origins[:,None]+np.arange(144)]
    records = []
    for name, value in [('absolute_hgb', base.values), ('daily_mean_state_ridge', values)]:
        for channel, err in [('load', value[:,:,0]-truth[:,:,0]),
            ('pv', value[:,:,1]-truth[:,:,1]),
            ('net', (value[:,:,0]-value[:,:,1])-(truth[:,:,0]-truth[:,:,1]))]:
            records.append({'name': name, 'channel': channel, **score(err, data.fixed_price)})
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    checks = []
    for index in (0, 14, 28, 59, 120, 240, 333):
        altered = data.actual.copy(); altered[int(base.origins[index]):] += [40000., 70000.]
        future = base.values.copy(); future[index+1:] += [90000., 20000.]
        b, _ = correct(altered, future, base.origins, index)
        np.testing.assert_array_equal(values[index], b)
        checks.append({'day': int(base.origins[index]//144), 'future_actual_and_forecasts_unchanged': True})
    np.testing.assert_array_equal(values[:,:,1], base.values[:,:,1])
    np.savez_compressed(OUT/'predictions.npz', values=values, origins=base.origins)
    save(OUT/'summary.json', {'complete': True, 'base_values_sha256': array_hash(base.values),
        'base_origins_sha256': array_hash(base.origins), 'metrics': records,
        'future_perturbation_checks': checks, 'pv_unchanged': True,
        'all_fit_labels_past': all(a.get('maximum_label_exclusive', 0)<=a['origin'] for a in audits)})
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
