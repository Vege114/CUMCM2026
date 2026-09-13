"""History-only PV amplitude transport of complete paired forecast errors.

One fixed empirical-scale transformation, with an unchanged 28-path control.
The positive-PV historical p95 is not a rated capacity or weather forecast.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.controller_candidate import INITIAL_SOC, plan_inventory, execute_inventory
from experiments.exp008.verify import verify_npz

OUT = Path('data/results/exp008/scaled_joint_risk')
SPEC = {'quantile': .8, 'controller': 'greedy', 'state_buffer': 500., 'wear': .002}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def pv_scale(actual, day):
    past = np.asarray(actual[max(0, day-28)*144:day*144, 1])
    positive = past[past > 0]
    return max(100., float(np.quantile(positive, .95))) if len(positive) else 100.


def risk_for_day(forecast, day):
    issue = forecast.get(day, scenario='2')
    history = forecast.net_error_paths(day, '2', limit=28)
    f = (issue['load_kw']-issue['pv_kw'])/6
    days = history['origins']//144
    old_scales = np.array([pv_scale(forecast.data.actual, int(d)) for d in days])
    current_scale = pv_scale(forecast.data.actual, day)
    ratio = current_scale/old_scales
    e = history['errors_kw']
    transported = (e[:, :, 0]-e[:, :, 1]*ratio[:, None])/6
    # Current means stay at the issued forecast: only the empirical error
    # sample is transformed, including its historical bias. No recentering.
    return {'base': f[None, :]+history['errors_kwh'],
            'scaled': f[None, :]+transported, 'forecast': f,
            'ratio': ratio, 'old_scales': old_scales, 'current_scale': current_scale,
            'origins': history['origins'], 'errors_kw': e,
            'audit': {'day': day, 'information_cutoff': day*144,
                      'max_observed_index': day*144-1,
                      'forecast': issue['audit'], 'history': history['audit'],
                      'current_pv_scale_kw': current_scale,
                      'historical_pv_scales_kw': old_scales.tolist(),
                      'historical_to_current_ratios': ratio.tolist(),
                      'risk_is_not_a_new_point_forecast': True}}


def crps(paths, truth):
    """Equal-weight empirical CRPS, last axis is the scenario sample."""
    x = np.sort(paths, axis=-1)
    count = x.shape[-1]
    weights = 2*np.arange(1, count+1)-count-1
    return np.mean(np.abs(x-truth[..., None]), axis=-1)-np.sum(x*weights, axis=-1)/count**2


def metrics(paths, truth, price):
    slot = crps(paths.transpose(0, 2, 1), truth)
    cumulative = crps(np.cumsum(paths, axis=2).transpose(0, 2, 1), np.cumsum(truth, axis=1))
    high = price >= np.quantile(price, .75)
    return {'slot_crps_kwh': float(slot.mean()),
            'high_price_slot_crps_kwh': float(slot[:, high].mean()),
            'prefix_energy_crps_kwh': float(cumulative.mean()),
            'daily_energy_crps_kwh': float(cumulative[:, -1].mean())}


def replay(records, forecast):
    directory = OUT/'scaled_point_lp'
    directory.mkdir()
    soc, mode = INITIAL_SOC, 1
    parts, rows, audits = [], [], []
    for day, record in zip(range(31, 365), records):
        risk = np.quantile(record['scaled'], .8, axis=0)
        plan = plan_inventory(np.repeat(risk[:, None], 9, axis=1), forecast.data.fixed_price,
                              soc, SPEC, final=day == 364)
        actual = forecast.data.actual[day*144:(day+1)*144, :2]
        detail, mode = execute_inventory(plan, actual, forecast.data.fixed_price, soc, mode, SPEC)
        parts.append(detail)
        audits.append(record['audit'])
        rows.append({'day': day, 'initial_soc': soc, 'final_soc': float(detail['states'][-1]),
                     'total_cost': float(detail['fees'].sum())})
        soc = rows[-1]['final_soc']
    arrays = {key: np.stack([p[key] for p in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 365)
    np.savez_compressed(directory/'dispatch_2.npz', **arrays)
    save(directory/'audit.json', audits)
    pd.DataFrame(rows).to_csv(directory/'daily.csv', index=False)
    checked = verify_npz(directory/'dispatch_2.npz', audit_path=directory/'audit.json')
    assert checked['passed'], checked['errors']
    save(directory/'summary.json', {'complete': True, 'verification': checked,
        'matched_control': 'cumulative_risk_planning/point', 'final_model_selection': False})
    return checked


def main():
    if OUT.exists():
        raise FileExistsError('Existing development results are immutable')
    OUT.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (OUT/'source_snapshot.py').write_bytes(source)
    save(OUT/'protocol.json', {'days': 334, 'one_fixed_candidate': True,
        'scale': 'max(100kW,p95 positivePV from preceding <=28 complete days)',
        'formula': 'currentNetForecast+(historicalLoadError-historicalPVError*currentScale/historicalIssueScale)/6',
        'sample_count': 28, 'january_fallback_labelled': True,
        'raw_joint_chronological_path_pairing_preserved': True,
        'no_current_day_truth_in_distribution_or_scale': True,
        'no_ratio_clipping_or_error_recentering': True,
        'no_rated_capacity_or_weather_interpretation': True,
        'matched_control': 'same complete28 raw errors in cumulative_risk_planning/point',
        'gate_before_lp': 'all four empirical CRPS metrics strictly decrease versus matched raw paths',
        'lp_spec': SPEC, 'true_empirical_lp_quantile': .8,
        'development_not_independent_test': True, 'source_sha256': hashlib.sha256(source).hexdigest()})
    forecast = AbsoluteHGBForecasts()
    input_hash = hashlib.sha256(forecast.store.values.tobytes()).hexdigest()
    records = [risk_for_day(forecast, day) for day in range(31, 365)]
    base = np.stack([r['base'] for r in records])
    scaled = np.stack([r['scaled'] for r in records])
    control = np.load('data/results/exp008/cumulative_risk_planning/risk_inputs.npz')
    np.testing.assert_allclose(base, control['forecast_net'][:, None, :]+control['errors'], rtol=0, atol=1e-10)
    mutation = []
    for day in (31, 32, 60, 151, 243, 364):
        changed = copy.copy(forecast.data)
        changed.actual = forecast.data.actual.copy()
        changed.actual[day*144:] += [70000., 30000., 1000.]
        after = AbsoluteHGBForecasts(changed)
        after.store.values[after.store.origins > day*144] += [60000., 20000.]
        r = risk_for_day(after, day)
        for key in ('base', 'scaled', 'forecast', 'ratio', 'old_scales', 'origins', 'errors_kw'):
            np.testing.assert_array_equal(r[key], records[day-31][key])
        assert r['current_scale'] == records[day-31]['current_scale']
        mutation.append({'day': day, 'future_actual_and_asymmetric_future_forecasts_unchanged': True})
    rng = np.random.default_rng(42)
    test = rng.normal(size=(5, 9)); truth_test = rng.normal(size=5)
    direct = np.mean(np.abs(test-truth_test[:, None]), axis=1)-.5*np.mean(np.abs(test[:, :, None]-test[:, None, :]), axis=(1, 2))
    np.testing.assert_allclose(crps(test, truth_test), direct, rtol=0, atol=1e-12)
    for r in records:
        e = r['errors_kw']
        np.testing.assert_allclose((e[:, :, 0]-e[:, :, 1]*np.ones((28, 1)))/6,
                                  r['base']-r['forecast'][None, :], atol=1e-10, rtol=0)
        assert np.all(r['ratio'] > 0) and np.all(r['origins']+144 <= r['audit']['information_cutoff'])
    np.savez_compressed(OUT/'risk_inputs.npz', base=base, scaled=scaled,
        ratios=np.stack([r['ratio'] for r in records]), current_scales=[r['current_scale'] for r in records],
        historical_scales=np.stack([r['old_scales'] for r in records]), days=np.arange(31, 365))
    save(OUT/'risk_audit.json', [r['audit'] for r in records])
    save(OUT/'verification.json', {'passed': True, 'future_mutations': mutation,
        'empirical_crps_matches_pairwise_formula': True,
        'unit_ratio_matches_original_error': True, 'matched_control_paths_identical': True,
        'issued_values_sha256_before': input_hash,
        'issued_values_sha256_after': hashlib.sha256(forecast.store.values.tobytes()).hexdigest()})
    # All methods are locked above; current truth is accessed here for scoring.
    actual = forecast.data.actual[31*144:, :2].reshape(334, 144, 2)
    truth = (actual[:, :, 0]-actual[:, :, 1])/6
    scores = {name: metrics(paths, truth, forecast.data.fixed_price)
              for name, paths in (('base', base), ('scaled', scaled))}
    gates = {key: scores['scaled'][key] < scores['base'][key] for key in scores['base']}
    result = {'complete': True, 'metrics': scores, 'gate_components': gates,
              'lp_continuation_gate_passed': all(gates.values()), 'final_model_selection': False}
    save(OUT/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if all(gates.values()):
        replay(records, forecast)


if __name__ == '__main__':
    main()
