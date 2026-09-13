"""Five conditional comparisons of forecast-shaped 24-dimensional purchases.

The saved baseline supplies each day's SOC, mode mask and historical paths.
All optimizations finish before current actuals are read for settlement.
The five candidate days do not form a continuous replacement policy.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.exp008.closed_loop import LIMIT, objective
from experiments.exp008.planner import ETA, execute
from experiments.exp008.verify import verify_arrays
from experiments.problem2.exp003.data import Data, ROOT

BASE = ROOT/'data/results/exp008/mode_planning_physical/load_memory_half_physical3_30days'
OUT = ROOT/'data/results/exp008/hourly_purchase_diagnostic'
OPTIONS = {'throughput': .002, 'variation': 0., 'terminal': .45, 'deadband': 0.}


def hourly_objective(delta, forecast_net, net_paths, price, initial_soc, charge_mask):
    unbounded = np.asarray(forecast_net)+np.repeat(delta, 6)
    purchase = np.maximum(unbounded, 0.)
    value, gradient = objective(purchase, net_paths, price, initial_soc,
                                charge_mask=charge_mask, **OPTIONS)
    return value, (gradient*(unbounded > 0)).reshape(-1, 6).sum(1)


def optimize_hourly(original, forecast_net, net_paths, price, initial_soc, charge_mask):
    if len(forecast_net) % 6:
        raise ValueError('horizon must contain complete hours')
    bound = LIMIT+float(np.max(np.abs(net_paths-forecast_net[None, :])))
    initial = np.clip((original-forecast_net).reshape(-1, 6).mean(1), -bound, bound)
    arguments = (forecast_net, net_paths, price, initial_soc, charge_mask)
    began = perf_counter()
    fit = minimize(hourly_objective, initial, args=arguments, jac=True, method='L-BFGS-B',
                   bounds=[(-bound, bound)]*len(initial),
                   options={'maxiter': 120, 'maxls': 30, 'ftol': 1e-9, 'gtol': 1e-5})
    purchase = np.maximum(0., forecast_net+np.repeat(fit.x, 6))
    return {'purchase': purchase, 'delta': fit.x, 'initial_delta': initial,
            'bound_kwh': bound,
            'initial_objective': float(hourly_objective(initial, *arguments)[0]),
            'final_objective': float(fit.fun), 'success': bool(fit.success),
            'message': str(fit.message), 'iterations': int(fit.nit),
            'evaluations': int(fit.nfev), 'seconds': perf_counter()-began,
            'global_optimality_certificate': False}


def gradient_check(delta, forecast_net, paths, price, soc, mask):
    args = (forecast_net, paths, price, soc, mask)
    _, analytic = hourly_objective(delta, *args)
    numeric = np.zeros_like(delta)
    epsilon = 1e-3
    for h in range(len(delta)):
        plus, minus = delta.copy(), delta.copy()
        plus[h] += epsilon
        minus[h] -= epsilon
        numeric[h] = (hourly_objective(plus, *args)[0]-hourly_objective(minus, *args)[0])/(2*epsilon)
    error = float(np.max(np.abs(numeric-analytic)))
    if error > 2e-4:
        raise AssertionError({'gradient_max_error': error, 'analytic': analytic.tolist(),
                              'numeric': numeric.tolist()})
    return {'passed': True, 'maximum_absolute_gradient_error': error, 'epsilon': epsilon}


def function_ast(path, name):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == name))


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    for module, function in [('closed_loop', 'objective'), ('planner', 'execute')]:
        assert function_ast(ROOT/f'experiments/exp008/{module}.py', function) == function_ast(
            BASE/f'{module}_snapshot.py', function)
    protocol = {'days': [31, 32, 33, 34, 35], 'base_directory': str(BASE),
                'comparison': 'same_initial_SOC_and_mask_per_day_not_continuous_replacement',
                'purchase': 'max(0, fixed_memory_forecast_net_kwh + delta_hour_kwh)',
                'dimensions': 24, 'initial_delta': 'hourly_mean(original_refined_q-forecast_net)',
                'bounds': 'plus/minus LIMIT+max(abs(historical_net_path-forecast_net))',
                'maxiter': 120, 'objective': OPTIONS,
                'current_actual_access': 'all five optimizations finish before settlement reads',
                'terminal_inventory_bound': 'positive(original_final_SOC-candidate_final_SOC)*ETA*5*max(price)',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'base_dispatch_sha256': hashlib.sha256((BASE/'dispatch.npz').read_bytes()).hexdigest(),
                'base_forecast_sha256': hashlib.sha256((BASE/'issued_memory_forecasts.npz').read_bytes()).hexdigest(),
                'known_kinks': 'zero-purchase point and clipped storage are nonsmooth; finite differences use interior probes',
                'full_policy_or_global_optimum_claimed': False}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    with np.load(BASE/'issued_memory_forecasts.npz') as f:
        forecast_origins, forecast_values = f['origins'].copy(), f['values'].copy()
    with np.load(BASE/'dispatch.npz') as saved:
        baseline_q = saved['original'].copy()
        saved_prices = saved['price'].copy()
        saved_masks = saved['allowed_charge'].copy()
    fitted = []
    for day in protocol['days']:
        i = day-31
        with np.load(BASE/f'planning_day{day}.npz') as p:
            paths, mask, price = p['all_net_paths'].copy(), p['allowed_charge'].copy(), p['price'].copy()
            soc, mode = float(p['initial_soc']), int(p['initial_mode'])
        assert paths.shape == (28, 144)
        np.testing.assert_array_equal(price, saved_prices[i])
        np.testing.assert_array_equal(mask, saved_masks[i])
        f = forecast_values[np.flatnonzero(forecast_origins == day*144)[0]]
        forecast_net = (f[:, 0]-f[:, 1])/6
        original = baseline_q[i]
        probe = (original-forecast_net).reshape(24, 6).mean(1)+np.linspace(.123, .731, 24)
        gradient = gradient_check(probe, forecast_net, paths, price, soc, mask)
        result = optimize_hourly(original, forecast_net, paths, price, soc, mask)
        old_objective = float(objective(original, paths, price, soc, charge_mask=mask, **OPTIONS)[0])
        np.savez_compressed(OUT/f'planning_day{day}.npz', original_purchase=original,
            purchase=result['purchase'], delta=result['delta'], initial_delta=result['initial_delta'],
            forecast_net=forecast_net, all_net_paths=paths, price=price, initial_soc=soc,
            initial_mode=mode, allowed_charge=mask)
        fitted.append({'day': day, 'initial_soc': soc, 'initial_mode': mode, 'price': price,
                       'mask': mask, 'original': original, 'result': result,
                       'original_historical_objective': old_objective, 'gradient': gradient})
        print('OPTIMIZED', day, 'history', old_objective, result['final_objective'], flush=True)
    # Only now do current-day actual arrays enter: solely execution/settlement.
    data = Data()
    rows, checks, audits = [], [], []
    with np.load(BASE/'dispatch.npz') as saved:
        for pair in fitted:
            day, i = pair['day'], pair['day']-31
            actual = data.actual[day*144:(day+1)*144]
            baseline_errors = {}
            details = {}
            previous_power = (172.76 if i == 0 else
                              6*float(saved['charge'][i-1, -1]-saved['discharge'][i-1, -1]))
            for name, q in [('original', pair['original']), ('hourly', pair['result']['purchase'])]:
                detail = execute(q, actual, pair['price'], pair['initial_soc'],
                                 charge_deadband=0., charge_mask=pair['mask'])
                detail.update(original=q, final=q.copy(), actual=actual.copy(), price=pair['price'])
                detail['fees'] = np.stack((q*pair['price'], np.zeros(144), np.zeros(144),
                                           5*detail['emergency']*pair['price']), axis=-1)
                arrays = {key: value[None] for key, value in detail.items()}
                check = verify_arrays(arrays, expected_days=1, start_day=day,
                    initial_soc=pair['initial_soc'], initial_mode=pair['initial_mode'],
                    initial_power_kw=previous_power, source_actual=actual[None], source_price=pair['price'])
                if not check['passed']:
                    raise AssertionError(check['errors'])
                checks.append({'day': day, 'name': name, 'verification': check})
                details[name] = detail
                np.savez_compressed(OUT/f'{name}_execution_day{day}.npz', **arrays)
                if name == 'original':
                    for key in ('charge', 'discharge', 'emergency', 'surplus', 'states', 'fees'):
                        error = float(np.max(np.abs(detail[key]-saved[key][i])))
                        baseline_errors[key] = error
                        if error > 1e-7:
                            raise AssertionError(baseline_errors)
            a, b = details['original'], details['hourly']
            cost_a, cost_b = float(a['fees'].sum()), float(b['fees'].sum())
            soc_a, soc_b = float(a['states'][-1]), float(b['states'][-1])
            compensation = max(0., soc_a-soc_b)*ETA*5*float(pair['price'].max())
            rows.append({'day': day, 'initial_soc': pair['initial_soc'],
                'original_historical_objective': pair['original_historical_objective'],
                'initial_hourly_historical_objective': pair['result']['initial_objective'],
                'optimized_hourly_historical_objective': pair['result']['final_objective'],
                'original_actual_cost': cost_a, 'hourly_actual_cost': cost_b,
                'actual_cost_change': cost_b-cost_a,
                'original_emergency_cost': float(a['fees'][:, 3].sum()),
                'hourly_emergency_cost': float(b['fees'][:, 3].sum()),
                'original_final_soc': soc_a, 'hourly_final_soc': soc_b,
                'lower_SOC_compensation_upper_bound': compensation,
                'conservative_savings': cost_a-cost_b-compensation})
            metadata = {k: v for k, v in pair['result'].items()
                        if k not in ('purchase', 'delta', 'initial_delta')}
            audits.append({'day': day, 'gradient_check': pair['gradient'],
                           'optimization': metadata, 'baseline_replay_errors': baseline_errors})
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'conditional_comparison.csv', index=False)
    conclusion = {'days': protocol['days'], 'conditional_not_continuous': True,
        'original_cost_sum': float(table.original_actual_cost.sum()),
        'hourly_cost_sum': float(table.hourly_actual_cost.sum()),
        'cost_change_sum': float(table.actual_cost_change.sum()),
        'lower_SOC_compensation_upper_bound_sum': float(table.lower_SOC_compensation_upper_bound.sum()),
        'conservative_savings_sum': float(table.conservative_savings.sum()),
        'days_with_conservative_savings': int(np.sum(table.conservative_savings > 0)),
        'all_gradient_and_physics_checks_passed': True,
        'all_original_baseline_arrays_reconstructed': True,
        'success_not_global_optimum': all(row['optimization']['success'] for row in audits)}
    (OUT/'audit.json').write_text(json.dumps(audits, indent=2))
    (OUT/'verification.json').write_text(json.dumps(checks, indent=2))
    (OUT/'summary.json').write_text(json.dumps(conclusion, indent=2))
    print(table.to_string(index=False), flush=True)
    print(json.dumps(conclusion, indent=2), flush=True)
    return conclusion


if __name__ == '__main__':
    run()
