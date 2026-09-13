"""Matched forecast-only bridge using the unchanged tree28/q.8/buffer500 LP.

This does not establish the full battery-action goal. Each call passes its
own frozen Store to risk_window.replay, including its historical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.load_energy_memory import EnergyMemoryStore, OUT as MEMORY_OUT
from experiments.exp008.neural_joint_calibration import JointStore
from experiments.exp008.risk_window import SPEC, replay, write_json
from experiments.exp008.verify import verify_npz
from experiments.problem2.exp003.data import Data, ROOT

OUT = MEMORY_OUT/'lp_bridge'


def run(name):
    if name not in ('joint', 'memory'):
        raise ValueError('two fixed pipelines only')
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data()
    store = EnergyMemoryStore(data=data) if name == 'memory' else JointStore(calibrated=True)
    case = f'{name}_tree28_q08_buffer500'
    directory = OUT/f'{case}_334days'
    directory.mkdir(exist_ok=True)
    protocol = {'name': name, 'days': 334, 'gain': .5 if name == 'memory' else None,
                'forecast': store.name, 'risk': 'tree28_built_from_this_store_current_and_prior_outputs',
                'planner': {**SPEC, 'calibration': store.name},
                'initial_soc': 1421.7991105135516, 'same_continuous_soc_start': True,
                'stage': 'forecast_only_fast_LP_bridge_not_final_action_constrained_policy',
                'comparison': 'full334day_predeclared_pair_not_selected_month_subset',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_data_hashes': data.hashes,
                'forecast_values_sha256': hashlib.sha256(store.values.tobytes()).hexdigest(),
                'dependencies': {}}
    dependencies = ('load_energy_memory', 'neural_joint_calibration', 'risk_window',
                    'controller_candidate', 'verify')
    for dependency in dependencies:
        path = ROOT/f'experiments/exp008/{dependency}.py'
        protocol['dependencies'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    old_protocol = directory/'bridge_protocol.json'
    if old_protocol.exists() and json.loads(old_protocol.read_text()) != protocol:
        raise RuntimeError('Refusing to reuse a changed bridge case')
    write_json(old_protocol, protocol)
    for dependency in dependencies:
        (directory/f'{dependency}_snapshot.py').write_bytes(
            (ROOT/f'experiments/exp008/{dependency}.py').read_bytes())
    (directory/'caller_snapshot.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(directory/'issued_forecasts.npz', origins=store.origins, values=store.values)
    result = replay(case, 28, 'tree', 334, data, store, OUT)
    # risk_window's mathematical code is unchanged. Correct its generic old
    # metadata labels in this new case so they name the actual supplied Store.
    audits = json.loads((directory/'audit.json').read_text())
    for audit in audits:
        audit['forecast_calibration'] = store.name
        audit['residual_source'] = ('periodic_baseline' if audit['fallback'] else
                                    f'{store.name}_prequential_forecast')
        if name == 'memory':
            audit['forecast_memory_audit'] = store.audit[store.lookup[audit['day']*144]]
    write_json(directory/'audit.json', audits)
    result['fixed_spec'] = {**SPEC, 'calibration': store.name}
    result['evaluation_role'] = protocol['stage']
    result['forecast_values_sha256'] = protocol['forecast_values_sha256']
    result['risk_window_generic_metadata_labels_corrected'] = True
    write_json(directory/'summary.json', result)
    check = verify_npz(directory/'dispatch_2.npz', expected_days=334, audit_path=directory/'audit.json')
    if not check['passed']:
        raise AssertionError(check['errors'])
    write_json(directory/'independent_verification.json', check)
    print(json.dumps({'name': name, 'cost': result['total_cost'],
                      'reversals': result['battery']['direction_reversals'],
                      'active_slots': result['battery']['active_slots'], 'verified': check['passed']}), flush=True)
    return result


def compare():
    frames, rows = {}, []
    for name in ('joint', 'memory'):
        directory = OUT/f'{name}_tree28_q08_buffer500_334days'
        if not (directory/'summary.json').exists():
            return
        frames[name] = pd.read_csv(directory/'daily.csv')
        result = json.loads((directory/'summary.json').read_text())
        rows.append({'name': name, 'total_cost': result['total_cost'],
                     'planned_cost': result['planned_cost'], 'emergency_cost': result['emergency_cost'],
                     **result['battery']})
    daily = frames['joint'][['day', 'date', 'total_cost', 'planned_cost', 'emergency_cost']].merge(
        frames['memory'][['day', 'total_cost', 'planned_cost', 'emergency_cost']],
        on='day', suffixes=('_joint', '_memory'))
    for column in ('total_cost', 'planned_cost', 'emergency_cost'):
        daily[column+'_change'] = daily[column+'_memory']-daily[column+'_joint']
    daily['month'] = pd.to_datetime(daily.date).dt.month
    monthly = daily.groupby('month').sum(numeric_only=True).drop(columns=['day'])
    daily.to_csv(OUT/'daily_comparison.csv', index=False)
    monthly.to_csv(OUT/'monthly_comparison.csv')
    pd.DataFrame(rows).to_csv(OUT/'annual_comparison.csv', index=False)
    write_json(OUT/'comparison.json', {'days': 334, 'fixed_gain': .5, 'rows': rows,
        'cost_change_yuan': rows[1]['total_cost']-rows[0]['total_cost'],
        'relative_change_pct': 100*(rows[1]['total_cost']/rows[0]['total_cost']-1),
        'not_a_final_action_target_certificate': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast', choices=('joint', 'memory', 'both'), default='both')
    arguments = parser.parse_args()
    for pipeline in ('joint', 'memory') if arguments.forecast == 'both' else (arguments.forecast,):
        run(pipeline)
    compare()
