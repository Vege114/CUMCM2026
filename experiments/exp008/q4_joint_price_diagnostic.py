"""Read-only price/net-risk association study; no new policy or annual solve.

Realized residual/emergency associations and bootstrap intervals are post-hoc
diagnostics. Separately, paired history samples are constructed only from
complete historical releases, showing what was available at each decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.issued_residual_paths import issued_error_paths
from experiments.exp008.unified_forecast import UnifiedForecasts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/exp008/q4_joint_price_diagnostic'
DATES = pd.date_range('2025-02-01', '2025-12-31')
SEASONS = {'spring_MarMay': (3, 4, 5), 'summer_JunAug': (6, 7, 8),
           'autumn_SepNov': (9, 10, 11), 'winter_FebDec_only': (2, 12)}
PRICE_FLOOR = .01


def corr(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    return float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-12 else None


def weighted_error(price_error, quantity):
    denominator = float(quantity.sum())
    return float(np.sum(price_error*quantity)/denominator) if denominator > 1e-12 else None


def day_bootstrap_weighted_error(price_error, quantity):
    """Whole days are resampled; intervals within a day remain together."""
    numerator, denominator = (price_error*quantity).sum(axis=1), quantity.sum(axis=1)
    ids = np.random.default_rng(20260913).integers(0, len(numerator), (1000, len(numerator)))
    divisor = denominator[ids].sum(axis=1)
    keep = divisor > 1e-12
    if not keep.any():
        return None
    value = numerator[ids].sum(axis=1)[keep]/divisor[keep]
    return np.quantile(value, [.025, .975]).tolist()


def association(price_error, net_error, emergency=None):
    positive = np.maximum(net_error, 0)
    mask = net_error > 0
    result = {
        'days': len(price_error), 'price_error_definition': 'actual minus issued forecast',
        'mean_price_error': float(price_error.mean()), 'price_error_mae': float(np.abs(price_error).mean()),
        'net_error_mae_kwh': float(np.abs(net_error).mean()),
        'corr_price_error_net_error': corr(price_error, net_error),
        'corr_price_error_positive_net_error': corr(price_error, positive),
        'within_slot_corr_price_error_net_error': corr(price_error-price_error.mean(0), net_error-net_error.mean(0)),
        'within_slot_corr_price_error_positive_net_error': corr(price_error-price_error.mean(0), positive-positive.mean(0)),
        'price_error_when_positive_net_error': float(price_error[mask].mean()) if mask.any() else None,
        'price_error_when_nonpositive_net_error': float(price_error[~mask].mean()) if (~mask).any() else None,
        'positive_net_error_weighted_price_error': weighted_error(price_error, positive),
        'positive_net_error_weighted_price_error_day_bootstrap_95pct': day_bootstrap_weighted_error(price_error, positive),
    }
    if emergency is not None:
        active = emergency > 1e-6
        result.update(
            emergency_kwh=float(emergency.sum()), emergency_slots=int(active.sum()),
            corr_price_error_emergency=corr(price_error, emergency),
            within_slot_corr_price_error_emergency=corr(price_error-price_error.mean(0), emergency-emergency.mean(0)),
            emergency_weighted_price_error=weighted_error(price_error, emergency),
            emergency_weighted_price_error_day_bootstrap_95pct=day_bootstrap_weighted_error(price_error, emergency),
            emergency_fee_actual_minus_issued_price_same_flows=float(5*np.sum(price_error*emergency)),
        )
    return result


def period_associations(price_error, net_error, emergency=None):
    result = {'annual': association(price_error, net_error, emergency), 'seasonal': {}, 'monthly': {}}
    for name, months in SEASONS.items():
        keep = np.isin(DATES.month, months)
        result['seasonal'][name] = association(price_error[keep], net_error[keep],
                                               None if emergency is None else emergency[keep])
    for month in sorted(set(DATES.month)):
        keep = DATES.month == month
        result['monthly'][str(month)] = association(price_error[keep], net_error[keep],
                                                    None if emergency is None else emergency[keep])
    return result


def collect(forecast, scenario):
    """Build issued predictions and legal paired supports without future truth."""
    midnight, latest, audits = [], [], []
    for day in range(31, 365):
        first = forecast.get(day, scenario=scenario)
        first_array = np.column_stack([first[k] for k in ('load_kw', 'pv_kw', 'price')])
        midnight.append(first_array)
        segments = []
        starts = (0, 36, 72, 108) if scenario == '4-3' else (0,)
        for start in starts:
            issue = first if start == 0 else forecast.get(day, start, scenario)
            length = 36 if scenario == '4-3' else 144
            segments.append(np.column_stack([issue[k][:length] for k in ('load_kw', 'pv_kw', 'price')]))
            history = (forecast.net_error_paths(day, scenario) if start == 0
                       else issued_error_paths(forecast, day, start, scenario))
            net_error = history['errors_kwh'][:, :length]
            price_error = history['price_errors'][:, :length]
            # p_j and net_j use the SAME historical day row. No actual value
            # from this target interval is consulted in this calculation.
            scenario_price = np.maximum(PRICE_FLOOR, issue['price'][None, :length]+price_error)
            positive = np.maximum(net_error, 0)
            paired = 5*np.sum(np.mean(scenario_price*positive, axis=0))
            independent = 5*np.sum(scenario_price.mean(0)*positive.mean(0))
            point = 5*np.sum(issue['price'][:length]*positive.mean(0))
            cutoff = day*144+start
            origins = np.asarray(history['origins'])
            assert np.all(origins+144 <= cutoff)
            assert history['audit']['max_observed_index'] < cutoff
            audits.append({
                'day': day, 'slot': start, 'information_cutoff': cutoff,
                'history_origins': origins.tolist(), 'history_count': len(origins),
                'max_observed_index': history['audit']['max_observed_index'],
                'paired_price_and_net_rows_same_origin': True,
                'forecast_calibration': forecast.calibration,
                'proxy': '5p times positive net forecast error; no battery or optimized emergency recourse',
                'paired_proxy_yuan': float(paired), 'independent_pairing_proxy_yuan': float(independent),
                'fixed_forecast_price_proxy_yuan': float(point),
                'pair_covariance_increment_yuan': float(paired-independent),
                'historical_mean_price_increment_yuan': float(independent-point),
                'clipped_price_samples': int(np.count_nonzero(scenario_price == PRICE_FLOOR)),
            })
        latest.append(np.concatenate(segments))
    return np.stack(midnight), np.stack(latest), audits


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = {'base': UnifiedForecasts(), 'ridge_28': UnifiedForecasts(calibration='ridge_28')}
    actual = forecasts['base'].data.actual[31*144:].reshape(334, 144, 3)
    results, predictions = {}, {}
    for calibration, forecast in forecasts.items():
        for scenario in ('4-2', '4-3'):
            key = f'{calibration}_{scenario}'
            midnight, latest, history_audits = collect(forecast, scenario)
            predictions[key] = (midnight, latest)
            net_error = ((actual[..., 0]-actual[..., 1])-(latest[..., 0]-latest[..., 1]))/6
            price_error = actual[..., 2]-latest[..., 2]
            history_frame = pd.DataFrame(history_audits)
            history_frame['month'] = DATES.month[np.asarray(history_frame['day'])-31]
            seasonal = {}
            for name, months in {'annual': tuple(range(1, 13)), **SEASONS}.items():
                rows = history_frame[np.isin(history_frame.month, months)]
                seasonal[name] = {
                    'releases': len(rows),
                    'paired_proxy_yuan': float(rows.paired_proxy_yuan.sum()),
                    'pair_covariance_increment_yuan': float(rows.pair_covariance_increment_yuan.sum()),
                    'historical_mean_price_increment_yuan': float(rows.historical_mean_price_increment_yuan.sum()),
                    'positive_pair_covariance_release_fraction': float((rows.pair_covariance_increment_yuan > 0).mean()),
                }
            results[key] = {
                'posthoc_realized_associations': period_associations(price_error, net_error),
                'issue_available_paired_history_positive_error_proxy': seasonal,
                'all_history_labels_complete': True,
                'clipped_price_sample_count': sum(row['clipped_price_samples'] for row in history_audits),
            }
            (OUT/f'paired_history_{key}.json').write_text(json.dumps(history_audits, ensure_ascii=False, indent=2))
    cases = []
    locations = [
        ('latest_forecast_old_dispatch', 'base'), ('joint_initial', 'base'),
        ('calibrated_quantile', 'ridge_28'),
        ('update_value_diagnostic/future1.5_d334_issued', 'base'),
    ]
    for case, calibration in locations:
        for scenario in ('4-2', '4-3'):
            path = ROOT/f'data/results/exp008/{case}/{scenario}/dispatch_{scenario}.npz'
            if not path.exists():
                continue
            with np.load(path) as z:
                if len(z['days']) != 334:
                    continue
                arrays = {key: z[key] for key in z.files}
            np.testing.assert_array_equal(arrays['actual'], actual)
            midnight, latest = predictions[f'{calibration}_{scenario}']
            price_error = actual[..., 2]-latest[..., 2]
            net_error = ((actual[..., 0]-actual[..., 1])-(latest[..., 0]-latest[..., 1]))/6
            fees = arrays['fees']
            q0, q = arrays['original'], arrays['final']
            planned_error = float(np.sum(q0*(actual[..., 2]-midnight[..., 2])))
            adjustment_error = float(np.sum((1.5*np.maximum(q-q0, 0)+.5*np.maximum(q0-q, 0))*price_error))
            emergency_error = float(np.sum(5*arrays['emergency']*price_error))
            cases.append({
                'case': case, 'scenario': scenario, 'forecast_calibration': calibration,
                'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'actual_total_cost': float(fees.sum()),
                'same_flows_bill_underestimation_yuan': planned_error+adjustment_error+emergency_error,
                'underestimation_components': {'planned_at_midnight': planned_error,
                                              'adjustment_at_latest_release': adjustment_error,
                                              'emergency_at_latest_release': emergency_error},
                'posthoc_realized_associations': period_associations(price_error, net_error, arrays['emergency']),
            })
    result = {
        'scope': 'read-only statistics of frozen forecasts and seven completed annual archives; no new policy optimization or annual replay',
        'residual_sign': 'actual minus issued forecast',
        'q4_3_alignment': 'each executed six-hour block uses its latest legal release; original purchase repricing uses midnight price',
        'historical_pairing': 'native net_error_paths at midnight; same-origin issued_residual_paths at intraday releases',
        'bootstrap': '1000 whole-day resamples, fixed seed; diagnostic interval only, not model selection',
        'historical_price_floor': PRICE_FLOOR,
        'season_note': 'winter contains February and December only because January is warmup',
        'forecasts': results, 'dispatch_cases': cases,
        'source_hashes': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                          for name in ('forecast.py', 'issued_residual_paths.py', 'q4_joint_price_diagnostic.py')},
    }
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('DONE', len(cases), 'complete archive diagnostics;', len(results), 'forecast/risk-pair diagnostics')
    return result


if __name__ == '__main__':
    run()
