"""Exploratory Q3/Q4-3 update-opportunity surrogate, with matched controls.

Only the next legally executed six-hour block has the true 5p emergency
penalty. Later blocks use a 1.5p opportunity proxy for future upward purchases.
This is not a multistage optimality certificate; realized bills always use the
unaltered problem settlement. Forecasts and physical execution are unchanged.
"""
import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.common.neural_v2.physics import settle

from .issued_residual_paths import issued_error_paths
from .planner import Settings, execute, plan
from .run import initial_state
from .unified_forecast import UnifiedForecasts
from .verify import battery_metrics, verify_npz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008/update_value_diagnostic'


def weights_for_release(slot, future_weight=1.5):
    if slot not in (0, 36, 72, 108):
        raise ValueError('Only the four legal information releases are allowed')
    weights = np.full(144-slot, float(future_weight))
    weights[:min(36, len(weights))] = 5.
    return weights


def run(scenario='3', future_weight=1.5, days=30, issued_residuals=False):
    if scenario not in ('3', '4-3'):
        raise ValueError('Update value applies only to Q3 and Q4-3')
    case = f'future{future_weight:g}_d{days}' + ('_issued' if issued_residuals else '')
    directory = OUT / case / scenario
    directory.mkdir(parents=True, exist_ok=True)
    config = {'scenario': scenario, 'future_weight': future_weight, 'days': days,
                  'issued_residuals': issued_residuals,
                  'settings': asdict(Settings()), 'calibration': None, 'deadband': 20., 'seed': 42,
                  'interpretation': 'exploratory later-update opportunity proxy; not a multistage certificate'}
    sources = ('planner.py', 'forecast.py', 'unified_forecast.py', 'run.py',
               'update_value_diagnostic.py', 'verify.py', 'issued_residual_paths.py')
    hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
              for name in sources}
    signature = hashlib.sha256(json.dumps([config, hashes], sort_keys=True).encode()).hexdigest()
    completion_path = directory/'completion.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion['signature'] != signature:
            raise RuntimeError(f'Refusing to overwrite changed experiment {directory}')
        return completion
    source_directory = directory/'source'
    source_directory.mkdir(exist_ok=True)
    for name in sources:
        shutil.copy2(Path(__file__).parent/name, source_directory/name)
    forecasts = UnifiedForecasts(seed=42, calibration=None)
    data = forecasts.data
    start_soc, start_power = initial_state(scenario)
    previous_power = start_power
    previous_mode = int(np.sign(previous_power))
    soc = start_soc
    parts, audits, daily = [], [], []
    began = perf_counter()
    for day in range(31, 31+days):
        states = np.empty(145)
        states[0] = soc
        components = {key: np.zeros(144) for key in ('charge', 'discharge', 'emergency', 'surplus')}
        original, final = None, np.zeros(144)
        for start in (0, 36, 72, 108):
            stop = start+36
            forecast = forecasts.get(day, slot=start, scenario=scenario)
            errors = (issued_error_paths(forecasts, day, start, scenario=scenario)
                      if issued_residuals else forecasts.net_error_paths(day, scenario=scenario))
            paths = errors['errors_kwh'] if isinstance(errors, dict) else errors
            weights = weights_for_release(start, future_weight)
            settings = SimpleNamespace(**{**asdict(Settings()), 'emergency_weight': weights})
            planned = plan(forecast, states[start], paths,
                           None if start == 0 else original[start:],
                           settings=settings, final_day=day == 364)
            purchase = planned['purchase']
            if start == 0:
                original = purchase.copy()
            final[start:] = purchase
            # Only the upcoming executed block is revealed after planning.
            actual_segment = data.actual[day*144+start:day*144+stop]
            price_segment = actual_segment[:, 2] if scenario.startswith('4') else data.fixed_price[start:stop]
            detail = execute(final[start:stop], actual_segment, price_segment, states[start],
                             charge_deadband=20., previous_power=previous_power,
                             previous_mode=previous_mode)
            for key, values in components.items():
                values[start:stop] = detail[key]
            states[start:stop+1] = detail['states']
            previous_power = float(6*(detail['charge'][-1]-detail['discharge'][-1]))
            signs = np.sign(detail['charge']-detail['discharge'])
            if np.any(signs):
                previous_mode = int(signs[signs != 0][-1])
            audits.append({'day': day, 'slot': start, 'information_cutoff': day*144+start,
                               'max_observed_index': errors['audit']['max_observed_index'],
                               'training_origins': np.asarray(errors['origins']).tolist(),
                               'residual_paths': errors['audit'],
                               'forecast': forecast['audit'], 'solver': planned['metadata'],
                               'planning_emergency_weights': weights.tolist(),
                               'original_locked': start == 0, 'executed_until': day*144+stop})
        actual = data.actual[day*144:(day+1)*144].copy()
        price = actual[:, 2].copy() if scenario.startswith('4') else data.fixed_price.copy()
        fees = settle(original, final, components['emergency'], price)
        parts.append(dict(original=original, final=final, **components, states=states,
                          fees=fees, actual=actual, price=price))
        soc = float(states[-1])
        daily.append({'day': day, 'planned_cost': float(fees[:, 0].sum()),
                          'up_cost': float(fees[:, 1].sum()), 'down_cost': float(fees[:, 2].sum()),
                          'emergency_cost': float(fees[:, 3].sum()), 'total_cost': float(fees.sum()),
                          'surplus_kwh': float(components['surplus'].sum()),
                          'throughput_kwh': float((components['charge']+components['discharge']).sum()),
                          'initial_soc': float(states[0]), 'final_soc': soc})
        if (day-30) % 30 == 0:
            print(case, scenario, day-30, sum(row['total_cost'] for row in daily), flush=True)
    archive = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    archive['days'] = np.arange(31, 31+days)
    archive_path = directory/f'dispatch_{scenario}.npz'
    np.savez_compressed(archive_path, **archive)
    pd.DataFrame(daily).to_csv(directory/'daily.csv', index=False)
    (directory/'audit.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2))
    verification = verify_npz(archive_path, scenario, expected_days=days,
                              initial_soc=start_soc, initial_power_kw=start_power,
                              initial_mode=int(np.sign(start_power)), audit_path=directory/'audit.json')
    (directory/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
    completion = {'config': config, 'source_hashes': hashes, 'signature': signature,
                      'complete': days == 334, 'exploratory': True, 'wall_seconds': perf_counter()-began,
                      'total_cost': float(archive['fees'].sum()), 'passed': verification['passed'],
                      'billing': verification['billing'], 'battery_metrics': verification['battery_metrics'],
                      'surplus_kwh': float(archive['surplus'].sum()), 'archive': str(archive_path)}
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2))
    if not verification['passed']:
        raise RuntimeError(verification['errors'])
    return completion


def compare(scenario, days, issued_residuals=False):
    results = [run(scenario, future_weight=weight, days=days,
                   issued_residuals=issued_residuals) for weight in (5., 1.5)]
    comparisons = {}
    for case in ('joint_initial', 'latest_forecast_old_dispatch'):
        with np.load(ROOT/f'data/results/exp008/{case}/{scenario}/dispatch_{scenario}.npz') as z:
            a = {key: z[key][:days] for key in z.files}
        comparisons[case] = {'total_cost': float(a['fees'].sum()),
                                 'fees_by_type': a['fees'].sum(axis=(0, 1)).tolist(),
                                 'surplus_kwh': float(a['surplus'].sum()), 'battery_metrics': battery_metrics(a)}
    result = {'scenario': scenario, 'days': days, 'exploratory': True, 'matched': results,
                  'prior_controls': comparisons,
                  'difference_candidate_minus_matched_control': results[1]['total_cost']-results[0]['total_cost']}
    suffix = '_issued' if issued_residuals else ''
    (OUT/f'comparison_{scenario}_d{days}{suffix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'matched'}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', default=['3'])
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--issued-residuals', action='store_true')
    parser.add_argument('--future-weights', nargs='+', type=float,
                        help='Run only specified arms; omit for a matched comparison')
    args = parser.parse_args()
    for scenario in args.scenarios:
        if args.future_weights is None:
            compare(scenario, args.days, args.issued_residuals)
        else:
            for weight in args.future_weights:
                run(scenario, future_weight=weight, days=args.days,
                    issued_residuals=args.issued_residuals)
