"""Predeclared 334-day fixed HGB/hold3 control evaluation; no performance gate."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil

import numpy as np
import pandas as pd
import scipy
import sklearn

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.closed_loop import optimize
from experiments.exp008.controller_candidate import INITIAL_SOC
from experiments.exp008.forecast_absolute_hgb import OUT as FORECAST_OUT, PRIMARY
from experiments.exp008.mode_minhold_physical import plan
from experiments.exp008.planner import execute
from experiments.exp008.verify import battery_metrics, verify_npz
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT/'data/results/exp008/mode_minhold_physical/direct_hgb_memory_hold3_physical3_334days'
HOURLY = ROOT/'data/results/exp008/mode_planning_physical/direct_hgb_memory_physical3_334days'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def prepare(forecast):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'evaluation_protocol.json').exists():
        raise FileExistsError('Predeclared evaluation already started; never silently reuse another prefix')
    protocol = {'evaluation_days': 334, 'candidate_count': 1, 'pipeline': PRIMARY,
        'evaluation_start_day': 31, 'evaluation_stop_exclusive': 365,
        'first3days_gate': 'feasible incumbent, all scenario physics, independent actual replay, gap<=.05',
        'first3day_cost_is_not_a_selection_gate': True, 'stop_from_partial_cost_or_count': False,
        'same_run_continues_after_first3_with_all_states': True,
        'mode_resolution_slots': 1, 'minimum_planned_mode_hold_slots': 3,
        'initial_soc': INITIAL_SOC, 'initial_planned_mode': 1, 'initial_planned_run_slots': 1,
        'initial_two_slots_locked': 'warmup final charge known; conservatively one completed charge slot',
        'mode_boundary': 'carry exact prior planned mode and completed run length; charge midnight switch fee',
        'planned_mode_is_not_same_as_nonidle_actual_action': True,
        'scenarios': 3, 'scenario_selection': 'equally spaced complete historical origins',
        'refinement_history_days': 28, 'refinement_maxiter': 120,
        'switching': 50., 'wear': .002, 'variation': 0., 'deadband': 0.,
        'mip_seconds': 5., 'target_mip_gap': .002,
        'terminal': 'original physical: price.min/ETA MIP and .45 greedy, both zero only day364',
        'actual_execution': 'strict_mask; observed current slot; continuous_SOC; exact bill only',
        'nonanticipative_recourse_certificate': False,
        'development_year_not_independent_test': True,
        'forecast_values_sha256': hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
        'forecast_origins_sha256': hashlib.sha256(forecast.store.origins.tobytes()).hexdigest()}
    save(OUT/'evaluation_protocol.json', protocol)
    # Freeze all local Python dependencies, including transitive imported
    # forecast helpers. Capturing a source file does not imply executing it.
    names = sorted({path for relative in ('experiments/exp008', 'experiments/common/neural_v2',
                    'experiments/problem2/exp003', 'experiments/problem2/exp004')
                    for path in (ROOT/relative).glob('*.py')})
    names += [path for path in (ROOT/'experiments/problem2/exp004/protocol.json', ROOT/'pyproject.toml', ROOT/'uv.lock') if path.exists()]
    source_hashes = {}
    for path in names:
        relative = path.relative_to(ROOT)
        target = OUT/'source_archive'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = digest(target)
    artifacts = sorted([path for path in FORECAST_OUT.iterdir() if path.is_file()]
                       + list((FORECAST_OUT/'models').glob('*.joblib')))
    forecast_hashes = {}
    for path in artifacts:
        relative = path.relative_to(FORECAST_OUT)
        target = OUT/'forecast_archive'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        forecast_hashes[str(relative)] = digest(target)
    np.savez_compressed(OUT/'issued_forecasts.npz', origins=forecast.store.origins, values=forecast.store.values)
    save(OUT/'complete_provenance.json', {'source_sha256': source_hashes,
        'forecast_artifact_sha256': forecast_hashes, 'source_data_sha256': forecast.data.hashes,
        'issued_archive_sha256': digest(OUT/'issued_forecasts.npz'), 'primary': PRIMARY,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
        'python': platform.python_version(), 'platform': platform.platform(),
        'model_training_not_repeated': True, 'no_other_forecast_prefix_reused': True})
    return protocol


def checkpoint(parts, audits, rows):
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 31+len(parts))
    np.savez_compressed(OUT/'dispatch.npz', **arrays)
    save(OUT/'planning_audit.json', audits)
    pd.DataFrame(rows).to_csv(OUT/'daily.csv', index=False)
    return arrays


def run():
    from experiments.exp008.hgb_mode_minhold_audit import audit
    data, forecast = Data(), AbsoluteHGBForecasts()
    prepare(forecast)
    parts, audits, rows = [], [], []
    soc, mode, age = INITIAL_SOC, 1, 1
    for day in range(31, 365):
        issued = forecast.get(day, scenario='2')
        issued['audit']['load_method'] = 'direct_absolute_HGB_Ridge28_memory_half'
        errors = forecast.net_error_paths(day, '2', limit=28)
        paths = (issued['load_kw']-issued['pv_kw'])[None, :]/6+errors['errors_kwh']
        selected = np.linspace(0, len(paths)-1, 3).astype(int)
        try:
            initialized = plan(paths[selected], data.fixed_price, soc, mode, age, final=day == 364)
        except (RuntimeError, AssertionError) as exc:
            if parts:
                checkpoint(parts, audits, rows)
            save(OUT/'failure.json', {'day': day, 'completed_days': len(parts), 'error': str(exc),
                 'no_extra_time_or_parameter_changes': True, 'annual_run_complete': False})
            raise
        meta = initialized['metadata']
        np.savez_compressed(OUT/f'planning_day{day}.npz',
            **{key: value for key, value in initialized.items() if key != 'metadata'},
            all_net_paths=paths, selected_scenario_indices=selected, initial_soc=soc,
            previous_planned_mode=mode, previous_planned_run_slots=age, price=data.fixed_price)
        refined = optimize(initialized['purchase'], paths, data.fixed_price, soc,
            charge_mask=initialized['allowed_charge'], throughput=.002, variation=0.,
            terminal=0. if day == 364 else .45, deadband=0., maxiter=120)
        q = refined['purchase']
        actual = data.actual[day*144:(day+1)*144]
        detail = execute(q, actual, data.fixed_price, soc, charge_mask=initialized['allowed_charge'], charge_deadband=0.)
        detail.update(original=q, final=q.copy(), actual=actual.copy(), price=data.fixed_price.copy(),
                      allowed_charge=initialized['allowed_charge'].copy())
        detail['fees'] = np.stack((q*data.fixed_price, np.zeros(144), np.zeros(144),
                                   5*detail['emergency']*data.fixed_price), axis=-1)
        parts.append(detail)
        audits.append({'day': day, 'forecast': issued['audit'], 'history': errors['audit'],
            'mip': meta, 'greedy_refinement': refined['metadata'],
            'planned_boundary_state_before': [mode, age],
            'planned_boundary_state_after': [meta['final_planned_mode'], meta['final_planned_run_slots']]})
        rows.append({'day': day, 'date': str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
            'total_cost': float(detail['fees'].sum()), 'emergency_cost': float(detail['fees'][:, 3].sum()),
            'initial_soc': soc, 'final_soc': float(detail['states'][-1]), 'mip_gap': meta['mip_gap'],
            'mip_seconds': meta['seconds'], 'planned_changes': meta['planned_changes_including_boundary'],
            'next_day_pending_mode_hold': meta['next_day_required_hold_slots']})
        soc, mode, age = rows[-1]['final_soc'], meta['final_planned_mode'], meta['final_planned_run_slots']
        print('DAY', day, 'cost', rows[-1]['total_cost'], 'gap', meta['mip_gap'], 'mode_state', mode, age, flush=True)
        if len(parts) in (3, 334) or len(parts) % 10 == 0:
            arrays = checkpoint(parts, audits, rows)
        if len(parts) == 3:
            checked = audit(expected_days=3)
            gate = {'passed': checked['passed'] and all(row['mip_gap'] is not None and row['mip_gap'] <= .05 for row in rows),
                'cost_and_count_not_used': True, 'days': 3, 'gaps': [row['mip_gap'] for row in rows],
                'independent_audit_sha256': digest(OUT/'independent_audit.json')}
            save(OUT/'feasibility_gate.json', gate)
            shutil.copy2(OUT/'independent_audit.json', OUT/'first3_independent_audit.json')
            np.savez_compressed(OUT/'first3_dispatch.npz', **arrays)
            if not gate['passed']:
                raise AssertionError('Predeclared computational/physics gate failed')
            print('FIRST3_FEASIBILITY_PASSED; CONTINUE_FIXED334', flush=True)
    verification = verify_npz(OUT/'dispatch.npz', expected_days=334, audit_path=OUT/'planning_audit.json')
    assert verification['passed'], verification['errors']
    save(OUT/'summary.json', {'completed_days': 334, 'total_cost': float(arrays['fees'].sum()),
        'battery': battery_metrics(arrays), 'verification': verification, 'last_planned_mode': mode,
        'last_planned_run_slots': age, 'next_day_mode_hold_remaining': max(0, 3-age),
        'nonanticipative_recourse_certificate': False, 'same_forecast_full_year': True})
    return audit(expected_days=334)


if __name__ == '__main__':
    run()
