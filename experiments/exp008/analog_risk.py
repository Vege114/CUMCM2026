"""One fixed causal analogue-day risk experiment; no forecast model changes.

Match whole issued residual days using information available at each day's
own midnight. Selected paths remain intact and equally weighted; the existing
inventory LP consumes their nine per-slot quantiles, not a joint scenario tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.controller_candidate import INITIAL_SOC, execute_inventory, plan_inventory
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.exp008.risk_window import SPEC, distribution_metrics
from experiments.exp008.risk_window import replay as replay_tree_control
from experiments.exp008.verify import verify_arrays
from experiments.problem2.exp003.data import Data
from experiments.problem2.tree_planning.risk import QUANTILE_LEVELS, TreeResidualScenarios

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/exp008/analog_risk'
FEATURE_NAMES = (
    'forecast_load_daily_mean_kw', 'forecast_pv_daily_mean_kw',
    'forecast_net_00_06_mean_kw', 'forecast_net_06_12_mean_kw',
    'forecast_net_12_18_mean_kw', 'forecast_net_18_24_mean_kw',
    'yesterday_pv_mean_minus_current_prediction_kw',
    'last3days_pv_mean_minus_current_prediction_kw', 'weekday_sin', 'weekday_cos',
)
HISTORY_DAYS, NEIGHBORS, HALF_LIFE_DAYS = 90, 21, 42.


def select_neighbors(historical_features, current_features, ages):
    """History-only scaling; Gaussian similarity times a fixed recency factor."""
    x, current, ages = map(np.asarray, (historical_features, current_features, ages))
    if x.ndim != 2 or current.shape != (x.shape[1],) or ages.shape != (len(x),) or not len(x):
        raise ValueError('Need nonempty history, one current feature vector, and history ages')
    if not np.isfinite(x).all() or not np.isfinite(current).all() or np.any(ages <= 0):
        raise ValueError('Features must be finite and historical ages must be positive')
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    distance = np.mean(((x-current)/scale)**2, axis=1)
    log_similarity = -.5*distance-np.log(2.)*ages/HALF_LIFE_DAYS
    selected = np.argsort(-log_similarity, kind='stable')[:min(NEIGHBORS, len(x))]
    # Preserve chronological row order after selecting complete days.
    return np.sort(selected), {
        'history_feature_mean': center.tolist(), 'history_feature_scale': scale.tolist(),
        'squared_standardized_distance': distance.tolist(),
        'log_similarity': log_similarity.tolist(),
    }


class AnalogResidualScenarios(TreeResidualScenarios):
    """Fixed 90-day pool / 21 analogues / 42-day half-life conditional paths."""

    def features_for_day(self, day):
        origin = int(day)*144
        if origin not in self._lookup or day < 3:
            raise ValueError('Features require a frozen midnight forecast and three prior days')
        prediction = self.values[self._lookup[origin]]
        # Historical features get their historical information cutoff, never
        # today's wider cutoff. Their own target-day actual is unavailable.
        previous = [self._completed_day(old, origin) for old in range(day-3, day)]
        load_mean, pv_mean = prediction.mean(axis=0)
        net_blocks = (prediction[:, 0]-prediction[:, 1]).reshape(4, 36).mean(axis=1)
        yesterday_pv = previous[-1][:, 1].mean()
        last3_pv = np.stack(previous)[:, :, 1].mean()
        weekday = (int(day)+2) % 7  # 2025-01-01 was Wednesday; calendar is known.
        angle = 2*np.pi*weekday/7
        return np.r_[load_mean, pv_mean, net_blocks, yesterday_pv-pv_mean,
                     last3_pv-pv_mean, np.sin(angle), np.cos(angle)]

    def path_bundle_for_day(self, day):
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer)):
            raise TypeError('day must be a zero-based integer')
        day = int(day)
        cutoff = day*144
        if cutoff not in self._lookup:
            raise ValueError('No frozen forecast at requested midnight')
        history = np.flatnonzero(self.origins+144 <= cutoff)[-HISTORY_DAYS:]
        if not len(history):
            # No issued ridge28 error exists on the first evaluation day.
            # Keep the known tree28 periodic fallback, explicitly not analogues.
            support, audit = super().for_day(day)
            audit.update(conditioning='analog_day', analog_fallback=True,
                         analog_fallback_reason='no_completed_issued_forecast_day',
                         selected_days=[], selected_count=0,
                         historical_feature_own_cutoff_enforced=True)
            return {'support': support, 'error_paths': np.empty((0, 144)), 'audit': audit}
        history_days = self.origins[history]//144
        x = np.stack([self.features_for_day(int(old)) for old in history_days])
        current = self.features_for_day(day)
        selected, scaling = select_neighbors(x, current, day-history_days)
        selected_days = history_days[selected]
        selected_ids = history[selected]
        errors = []
        for index, old in zip(selected_ids, selected_days):
            actual = self._completed_day(int(old), cutoff)
            prediction = self.values[index]
            errors.append(((actual[:, 0]-actual[:, 1])-
                           (prediction[:, 0]-prediction[:, 1]))/6)
        errors = np.stack(errors)
        prediction = self.values[self._lookup[cutoff]]
        current_net = (prediction[:, 0]-prediction[:, 1])/6
        support = current_net[:, None]+np.quantile(errors, QUANTILE_LEVELS, axis=0, method='linear').T
        audit = {
            'day': day, 'information_cutoff': cutoff, 'cutoff_is_exclusive': True,
            'max_observed_index': cutoff-1,
            'training_origins': self.origins[history].tolist(),
            'selected_days': selected_days.tolist(), 'selected_count': len(selected),
            'candidate_days': history_days.tolist(), 'candidate_count': len(history),
            'selected_feature_cutoffs': (selected_days*144).tolist(),
            'selected_label_stops_exclusive': ((selected_days+1)*144).tolist(),
            'historical_feature_own_cutoff_enforced': True,
            'feature_names': list(FEATURE_NAMES), 'current_features': current.tolist(),
            'historical_features': x.tolist(), **scaling,
            'history_days_limit': HISTORY_DAYS, 'neighbors_limit': NEIGHBORS,
            'time_half_life_days': HALF_LIFE_DAYS,
            'similarity_formula': 'exp(-0.5*mean(z_difference**2))*2**(-age_days/42)',
            'standardization_uses_current_features': False,
            'conditioning': 'analog_day', 'analog_fallback': False,
            'residual_source': 'ridge28_prequential_forecast',
            'path_weights': np.full(len(errors), 1/len(errors)).tolist(),
            'quantile_levels': QUANTILE_LEVELS.tolist(), 'weights': self.weights.tolist(),
            'scenario_semantics': 'whole_daily_errors_selected_then_slot_marginals_for_existing_lp',
        }
        return {'support': support, 'error_paths': errors, 'audit': audit}

    def for_day(self, day):
        bundle = self.path_bundle_for_day(day)
        return bundle['support'], bundle['audit']


class MatureAnalogResidualScenarios(AnalogResidualScenarios):
    """Use unchanged tree28 until 21 complete issued error days are available.

    The threshold is fixed for tail-support sample size, independent of bills.
    Once mature, all analogue features, scores and equal path weights remain
    exactly those of the original fixed experiment.
    """

    def path_bundle_for_day(self, day):
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer)):
            raise TypeError('day must be a zero-based integer')
        day = int(day)
        count = int(np.count_nonzero(self.origins+144 <= day*144))
        if count < NEIGHBORS:
            support, audit = TreeResidualScenarios.for_day(self, day)
            audit.update(conditioning='analog_day_warm21', analog_fallback=True,
                         analog_fallback_reason='fewer_than_21_completed_issued_days',
                         selected_days=[], selected_count=0, candidate_count=count,
                         min_issued_days_before_analog=NEIGHBORS,
                         historical_feature_own_cutoff_enforced=True)
            return {'support': support, 'error_paths': np.empty((0, 144)), 'audit': audit}
        result = super().path_bundle_for_day(day)
        result['audit']['min_issued_days_before_analog'] = NEIGHBORS
        return result


def run(days=30, out=OUT, mature_start=False):
    out = Path(out)
    suffix = '_warm21' if mature_start else ''
    directory = out/f'analog90_n21_half42{suffix}_{days}days'
    directory.mkdir(parents=True, exist_ok=True)
    data, store = Data(), CalibratedStore('ridge_28')
    risk_type = MatureAnalogResidualScenarios if mature_start else AnalogResidualScenarios
    risk = risk_type(store.origins, store.values, data.actual, data.fixed_price)
    config = {
        'fixed_spec': SPEC, 'days': days, 'history_days': HISTORY_DAYS,
        'neighbors': NEIGHBORS, 'time_half_life_days': HALF_LIFE_DAYS,
        'features': list(FEATURE_NAMES), 'only_one_candidate': True,
        'min_issued_days_before_analog': NEIGHBORS if mature_start else 0,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'controller_sha256': hashlib.sha256((Path(__file__).parent/'controller_candidate.py').read_bytes()).hexdigest(),
        'forecast_values_sha256': hashlib.sha256(store.values.tobytes()).hexdigest(),
    }
    completion = directory/'summary.json'
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved['config'] != config:
            raise RuntimeError('Refusing to overwrite changed analogue experiment')
        return saved
    source_directory = directory/'source'
    source_directory.mkdir(exist_ok=True)
    for name in ('analog_risk.py', 'controller_candidate.py', 'risk_window.py'):
        shutil.copy2(Path(__file__).parent/name, source_directory/name)
    soc, mode, parts, rows, audits, supports, paths, counts = INITIAL_SOC, 1, [], [], [], [], [], []
    began = perf_counter()
    for day in range(31, 31+days):
        bundle = risk.path_bundle_for_day(day)
        support, audit, error_paths = bundle['support'], bundle['audit'], bundle['error_paths']
        plan = plan_inventory(support, data.fixed_price, soc, SPEC, final=day == 364)
        observed = data.actual[day*144:(day+1)*144]
        detail, mode = execute_inventory(plan, observed, data.fixed_price, soc, mode, SPEC)
        audit.update(forecast_origin=day*144, purchase_locked_before_actual_read=True)
        parts.append(detail)
        audits.append(audit)
        supports.append(support)
        padded = np.zeros((NEIGHBORS, 144))
        padded[:len(error_paths)] = error_paths
        paths.append(padded)
        counts.append(len(error_paths))
        rows.append({'day': day, 'total_cost': float(detail['fees'].sum()),
                     'planned_cost': float(detail['fees'][:, 0].sum()),
                     'emergency_cost': float(detail['fees'][:, 3].sum()),
                     'initial_soc': soc, 'final_soc': float(detail['states'][-1])})
        soc = rows[-1]['final_soc']
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 31+days)
    supports = np.stack(supports)
    truth = (arrays['actual'][..., 0]-arrays['actual'][..., 1])/6
    verified = verify_arrays(arrays, expected_days=days,
                             source_actual=data.actual[31*144:(31+days)*144].reshape(days, 144, 2),
                             source_price=data.fixed_price, audit_records=audits)
    if not verified['passed']:
        raise AssertionError(verified['errors'])
    result = {
        'config': config, **verified['billing'], 'battery': verified['battery_metrics'],
        'distribution': distribution_metrics(supports, truth, data.fixed_price),
        'verified': True, 'elapsed_seconds': perf_counter()-began,
        'evaluation_role': 'single fixed 2025 development experiment, not untouched validation',
    }
    np.savez_compressed(directory/'dispatch_2.npz', **arrays)
    np.savez_compressed(directory/'supports.npz', supports=supports,
                        selected_error_paths=np.stack(paths), selected_path_counts=np.asarray(counts))
    pd.DataFrame(rows).to_csv(directory/'daily.csv', index=False)
    for name, value in [('audit.json', audits), ('verification.json', verified), ('summary.json', result)]:
        (directory/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    if days == 30 or mature_start:
        if mature_start:
            baseline = replay_tree_control('tree28', 28, 'tree', days, data, store, out)
            with np.load(out/f'tree28_{days}days/dispatch_2.npz') as reference:
                prefix = min(21, days)
                prefix_differences = {key: float(np.max(np.abs(value[:prefix]-reference[key][:prefix])))
                                      for key, value in arrays.items()}
            with np.load(out/f'tree28_{days}days/supports.npz') as reference:
                prefix_support_error = float(np.max(np.abs(supports[:prefix]-reference['supports'][:prefix])))
            if any(value != 0 for value in prefix_differences.values()) or prefix_support_error != 0:
                raise AssertionError('Warmup prefix must exactly match tree28')
        else:
            baseline = json.loads((ROOT/'data/results/exp008/risk_window/tree28_30days/summary.json').read_text())
        depletion = max(0, baseline['battery']['final_soc']-result['battery']['final_soc'])*np.sqrt(.9)*5*data.fixed_price.max()
        saving = baseline['total_cost']-result['total_cost']-depletion
        comparison = {
            'fixed_baseline': f'ridge28/tree28/q0.8/state_buffer500; identical {days} days',
            'baseline': baseline, 'candidate': result,
            'delta_cost': result['total_cost']-baseline['total_cost'],
            'final_soc_depletion_upper_bound_cost': float(depletion),
            'conservative_saving': float(saving),
            'predeclared_meaningful_improvement_gate': 'at least 0.5% saving after conservative final-SOC adjustment',
            'meaningful_improvement': bool(saving >= .005*baseline['total_cost']),
            'no_automatic_annual_extension': True,
        }
        if mature_start:
            comparison.update(min_complete_issued_days=NEIGHBORS,
                              first_analog_day=31+NEIGHBORS,
                              analog_active_decision_days=max(0, days-NEIGHBORS),
                              prefix_maximum_absolute_differences=prefix_differences,
                              prefix_support_maximum_difference=prefix_support_error,
                              first_21_decision_days_exact_tree28=True,
                              sufficient_history_conservative_cost_advantage=bool(saving > 0))
        comparison_name = f'warm21_comparison_{days}days.json' if mature_start else 'pilot_comparison.json'
        (out/comparison_name).write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in comparison.items() if k not in ('baseline', 'candidate')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--mature-start', action='store_true',
                        help='Use original tree28 until 21 complete issued days; one fixed cold-start repair')
    args = parser.parse_args()
    run(args.days, mature_start=args.mature_start)
