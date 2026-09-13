"""One fixed 30-day paired-price experiment against frozen Q4-2 physical.

Only optimization prices change. Current actual prices enter after physical
execution for settlement. No automatic annual extension or parameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.joint_price_closed_loop import optimize
from experiments.exp008.joint_price_physical import plan
from experiments.exp008.planner import execute
from experiments.exp008.unified_forecast import UnifiedForecasts
from experiments.exp008.verify import verify_arrays

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/exp008/q4_joint_price'
CONFIG = {
    'days': 30, 'scenario': '4-2', 'calibration': 'ridge_28', 'initialization_scenarios': 3,
    'refinement_history_days': 28, 'block_slots': 6, 'switching': 50., 'wear': .002,
    'seconds': 5., 'gap': .002, 'refinement_maxiter': 120, 'deadband': 0., 'variation': 0.,
    'terminal_initialization': 'minimum_current_forecast_price / ETA',
    'terminal_refinement': .45, 'price_floor': .01,
    'price_and_net_pairing': 'identical historical issued day rows and identical selected indices',
    'baseline': 'q4_physical/pilot30',
    'gate': 'strictly lower true 30-day bill and fewer non-idle reversals than frozen physical control',
    'no_automatic_annual_extension': True,
}


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def planning_inputs(forecast, day):
    issue = forecast.get(day, scenario='4-2')
    history = forecast.net_error_paths(day, '4-2', limit=28)
    net = (issue['load_kw']-issue['pv_kw'])[None, :]/6+history['errors_kwh']
    prices = np.maximum(.01, issue['price'][None, :]+history['price_errors'])
    selected = np.linspace(0, len(net)-1, min(3, len(net))).astype(int)
    assert np.all(history['origins']+144 <= day*144)
    assert history['audit']['max_observed_index'] < day*144
    return {'net_paths': net, 'scenario_prices': prices, 'selected_indices': selected,
            'predicted_price': issue['price'], 'forecast_audit': issue['audit'],
            'history_audit': history['audit'], 'history_origins': history['origins'],
            'clipped_price_samples': int(np.count_nonzero(prices == .01))}


def run(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = out/'daily_chunks'
    chunks.mkdir(exist_ok=True)
    forecast = UnifiedForecasts(calibration='ridge_28')
    data = forecast.data
    warmup_path = ROOT/'data/results/exp002/warmup_4-2.npz'
    warmup = read_npz(warmup_path)
    initial_soc = float(warmup['states'][-1, -1])
    initial_power = float(6*(warmup['charge'][-1, -1]-warmup['discharge'][-1, -1]))
    initial_mode = int(np.sign(initial_power))
    baseline_path = ROOT/'data/results/exp008/q4_physical/pilot30/dispatch_4-2.npz'
    baseline = read_npz(baseline_path)
    names = ('run_q4_joint_price', 'joint_price_physical', 'joint_price_closed_loop',
             'planner', 'unified_forecast', 'forecast', 'forecast_calibration', 'verify')
    files = [ROOT/f'experiments/exp008/{name}.py' for name in names]
    protocol = {
        'config': CONFIG, 'initial_soc': initial_soc, 'initial_mode': initial_mode,
        'initial_power_kw': initial_power, 'source_data_hashes': data.hashes,
        'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'ridge28_values_sha256': hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'copied_physical_constraints_source_sha256': hashlib.sha256((Path(__file__).parent/'mode_planning_physical.py').read_bytes()).hexdigest(),
        'copied_closed_loop_source_sha256': hashlib.sha256((Path(__file__).parent/'closed_loop.py').read_bytes()).hexdigest(),
    }
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise RuntimeError('Changed source/configuration: choose a new output directory')
    else:
        write_json(out/'protocol.json', protocol)
        snapshots = out/'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for path in files:
            (snapshots/path.name).write_bytes(path.read_bytes())
    if (out/'pilot30/summary.json').exists():
        return json.loads((out/'pilot30/summary.json').read_text())
    soc, mode, parts, audits = initial_soc, initial_mode, [], []
    began = perf_counter()
    for day in range(31, 61):
        archive_path, audit_path = chunks/f'day_{day}.npz', chunks/f'day_{day}.json'
        if archive_path.exists() and audit_path.exists():
            detail, audit = read_npz(archive_path), json.loads(audit_path.read_text())
            assert abs(float(detail['states'][0])-soc) < 1e-6
        else:
            inputs = planning_inputs(forecast, day)
            selected = inputs['selected_indices']
            initialized = plan(inputs['net_paths'][selected], inputs['predicted_price'], soc, mode,
                               scenario_prices=inputs['scenario_prices'][selected],
                               block_slots=6, switching=50., wear=.002, seconds=5., gap=.002)
            refined = optimize(initialized['purchase'], inputs['net_paths'], inputs['scenario_prices'], soc,
                               charge_mask=initialized['allowed_charge'], throughput=.002,
                               variation=0., terminal=.45, maxiter=120, deadband=0.)
            q = refined['purchase']
            # Observation is revealed only after both optimization stages.
            actual_load_pv = data.actual[day*144:(day+1)*144, :2]
            detail = execute(q, actual_load_pv, inputs['predicted_price'], soc,
                             charge_deadband=0., charge_mask=initialized['allowed_charge'])
            actual = data.actual[day*144:(day+1)*144].copy()
            actual_price = actual[:, 2]
            detail.update(original=q, final=q.copy(), actual=actual, price=actual_price.copy(),
                          allowed_charge=initialized['allowed_charge'].copy())
            detail['fees'] = np.stack((q*actual_price, np.zeros(144), np.zeros(144),
                                        5*detail['emergency']*actual_price), axis=-1)
            audit = {
                'day': day, 'information_cutoff': day*144, 'forecast': inputs['forecast_audit'],
                'history': inputs['history_audit'], 'mip': initialized['metadata'],
                'refinement': refined['metadata'], 'initial_soc': soc, 'initial_mode': mode,
                'training_origins': inputs['history_origins'].tolist(),
                'selected_history_origins': inputs['history_origins'][selected].tolist(),
                'paired_scenario_prices_and_net': True,
                'clipped_price_samples': inputs['clipped_price_samples'],
                'actual_price_used_only_in_settlement': True,
            }
            np.savez_compressed(chunks/f'planning_day_{day}.npz',
                                all_net_paths=inputs['net_paths'], all_scenario_prices=inputs['scenario_prices'],
                                selected_indices=selected, predicted_price=inputs['predicted_price'],
                                initial_purchase=initialized['purchase'], refined_purchase=q,
                                allowed_charge=initialized['allowed_charge'],
                                **{key: initialized[key] for key in ('scenario_charge', 'scenario_discharge',
                                    'scenario_emergency', 'scenario_surplus', 'scenario_states')})
            np.savez_compressed(archive_path, **detail)
            write_json(audit_path, audit)
        parts.append(detail)
        audits.append(audit)
        soc = float(detail['states'][-1])
        signs = np.sign(detail['charge']-detail['discharge'])
        if np.any(signs):
            mode = int(signs[signs != 0][-1])
        if len(parts) % 5 == 0:
            print('PROGRESS', len(parts), sum(float(x['fees'].sum()) for x in parts),
                  'seconds', perf_counter()-began, flush=True)
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 61)
    actual = data.actual[31*144:61*144].reshape(30, 144, 3)
    kwargs = {'scenario': '4-2', 'expected_days': 30, 'initial_soc': initial_soc,
              'initial_mode': initial_mode, 'initial_power_kw': initial_power,
              'source_actual': actual, 'source_price': actual[..., 2]}
    verification = verify_arrays(arrays, audit_records=audits, **kwargs)
    baseline_check = verify_arrays(baseline, **kwargs)
    assert verification['passed'], verification['errors']
    assert baseline_check['passed'], baseline_check['errors']
    total, reference = verification['billing']['total_cost'], baseline_check['billing']['total_cost']
    metric, reference_metric = verification['battery_metrics'], baseline_check['battery_metrics']
    gaps = [audit['mip']['mip_gap'] for audit in audits]
    summary = {
        'config': CONFIG, **verification['billing'], 'battery': metric,
        'baseline_billing': baseline_check['billing'], 'baseline_battery': reference_metric,
        'delta_cost': total-reference, 'cost_reduction_pct': 100*(1-total/reference),
        'reversal_delta': metric['direction_reversals']-reference_metric['direction_reversals'],
        'cost_and_reversals_gate': bool(total < reference and metric['direction_reversals'] < reference_metric['direction_reversals']),
        'verified': True, 'mip_gaps': gaps, 'seconds': perf_counter()-began,
        'all_scenario_physics_passed': all(a['mip']['scenario_physical_checks']['passed'] for a in audits),
        'nonanticipative_recourse_certificate': False, 'global_optimality_certificate': False,
        'actual_prices_used_only_after_execution': True,
        'no_automatic_annual_extension': True,
    }
    directory = out/'pilot30'
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/'dispatch_4-2.npz', **arrays)
    pd.DataFrame([{'day': 31+i, 'total_cost': float(d['fees'].sum()),
                   'planned_cost': float(d['fees'][:, 0].sum()), 'emergency_cost': float(d['fees'][:, 3].sum()),
                   'initial_soc': float(d['states'][0]), 'final_soc': float(d['states'][-1])}
                  for i, d in enumerate(parts)]).to_csv(directory/'daily.csv', index=False)
    for name, value in [('summary.json', summary), ('verification.json', verification),
                         ('baseline_verification.json', baseline_check), ('audit.json', audits)]:
        write_json(directory/name, value)
    print('DONE', json.dumps({key: summary[key] for key in ('total_cost', 'delta_cost', 'reversal_delta',
                                                          'cost_and_reversals_gate')}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)
