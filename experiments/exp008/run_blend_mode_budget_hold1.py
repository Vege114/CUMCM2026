"""Predeclared fixed raw HGB/ExtraTrees blend; unchanged hold1 physical cap8 policy."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from experiments.exp008.forecast import Forecasts
from experiments.exp008.forecast_hgb_extra_trees_half import HGBExtraTreesHalfStore, HGB, EXTRA
from experiments.exp008.closed_loop import optimize
from experiments.exp008.controller_candidate import INITIAL_SOC
from experiments.exp008.forecast_hgb_extra_trees_half import OUT as FORECAST_OUT, PRIMARY
from experiments.exp008.mode_budget_hold1_physical import plan
from experiments.exp008.planner import execute
from experiments.exp008.verify import battery_metrics, verify_npz
from experiments.problem2.exp003.data import ROOT, Data

OUT = ROOT/'data/results/exp008/mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
HGB_HOLD1 = ROOT/'data/results/exp008/mode_budget_hold1_physical/direct_hgb_memory_hold1_cap8_kappa0_334days'


class BlendForecasts(Forecasts):
    def __init__(self, data=None):
        super().__init__(data=data, seed=42)
        self.store = HGBExtraTreesHalfStore()
        self.calibration = PRIMARY

    def get(self, day, slot=0, scenario='2'):
        result = super().get(day, slot, scenario)
        result['audit'].update(
            selected_model_id=PRIMARY,
            base_forecast='fixed_raw_HGB_ExtraTrees_half' if day >= 31 else 'periodic_cold_start',
            load_method=PRIMARY if day >= 31 else 'periodic_cold_start',
            output_calibration='same_Ridge28_then_nonrecursive_half_load_memory' if day >= 31 else None)
        return result

    def net_error_paths(self, day, scenario='2', limit=28):
        result = super().net_error_paths(day, scenario, limit)
        result['audit'].update(model_id=PRIMARY,
            source='same_fixed_raw_blend_Ridge28_memory_with_labelled_January_periodic_cold_start')
        return result


def local_dependency_paths():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(ROOT/'experiments') and path.suffix == '.py':
                paths.add(path)
    return sorted(paths)


def source_consistency():
    provenance = json.loads((OUT/'complete_provenance.json').read_text())
    for path in local_dependency_paths():
        relative = str(path.relative_to(ROOT))
        assert relative in provenance['source_sha256'], relative
    for relative, expected in provenance['source_sha256'].items():
        assert digest(ROOT/relative) == expected, relative
        assert digest(OUT/'source_archive'/relative) == expected, relative
    for source, expected in provenance['forecast_original_sha256'].items():
        assert digest(Path(source)) == expected, source
    return {'passed':True, 'all_actual_local_imports_frozen_before_run':True,
            'source_files':len(provenance['source_sha256']),
            'all_live_sources_and_model_inputs_unchanged':True,
            'tree_planning_transitive_helpers_included':True}


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
        'mode_resolution_slots': 1, 'minimum_planned_mode_hold_slots': 1,
        'initial_soc': INITIAL_SOC, 'initial_planned_mode': 1, 'initial_planned_run_slots': 1,
        'initial_two_slots_locked': False,
        'change_from_original_HGB_hold1': 'only issued forecast pipeline and its own issued residual paths change',
        'original_hold1_auxiliary_relaxation_retained': 'minimum10minutes, initial two-slot lock absent',
        'physical_battery_parameters_unchanged': True,
        'intensity_reporting': ['throughput', 'EFC', 'active_slots', 'power_TV',
            'planned_1_2_3_slot_runs', 'contiguous_actual_active_1_2_3_slot_runs'],
        'switching_budget_does_not_prove_lifetime_improvement': True,
        'mode_boundary': 'carry exact prior planned mode and run length; count midnight switch against daily cap',
        'planned_mode_is_not_same_as_nonidle_actual_action': True,
        'scenarios': 3, 'scenario_selection': 'equally spaced complete historical origins',
        'refinement_history_days': 28, 'refinement_maxiter': 120,
        'switching': 0., 'daily_planned_switch_cap': 8, 'annual_switch_upper_bound': 2672,
        'budget_includes_midnight_boundary': True, 'flip_is_exact_binary_XOR': True, 'wear': .002, 'variation': 0., 'deadband': 0.,
        'mip_seconds': 5., 'target_mip_gap': .002,
        'terminal': 'original physical: price.min/ETA MIP and .45 greedy, both zero only day364',
        'actual_execution': 'strict_mask; observed current slot; continuous_SOC; exact bill only',
        'nonanticipative_recourse_certificate': False,
        'development_year_not_independent_test': True,
        'forecast_values_sha256': hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
        'forecast_origins_sha256': hashlib.sha256(forecast.store.origins.tobytes()).hexdigest()}
    save(OUT/'evaluation_protocol.json', protocol)
    # Actual loaded local dependency closure, including transitive tree_planning.
    # All audit imports are loaded before prepare; no glob capture of unrelated
    # concurrent agents' sources, and all live files are rechecked at both ends.
    names = local_dependency_paths()
    names += [path for path in (ROOT/'experiments/problem2/exp004/protocol.json', ROOT/'pyproject.toml', ROOT/'uv.lock') if path.exists()]
    source_hashes = {}
    for path in names:
        relative = path.relative_to(ROOT)
        target = OUT/'source_archive'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = digest(target)
    forecast_hashes, original_hashes = {}, {}
    for directory in (FORECAST_OUT, HGB, EXTRA):
        artifacts = sorted([path for path in directory.iterdir() if path.is_file()]
                           + list((directory/'models').glob('*.joblib')))
        for path in artifacts:
            relative = Path(directory.name)/path.relative_to(directory)
            target = OUT/'forecast_archive'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            forecast_hashes[str(relative)] = digest(target)
            original_hashes[str(path)] = digest(path)
    np.savez_compressed(OUT/'issued_forecasts.npz', origins=forecast.store.origins, values=forecast.store.values)
    save(OUT/'complete_provenance.json', {'source_sha256': source_hashes,
        'forecast_artifact_sha256': forecast_hashes, 'forecast_original_sha256': original_hashes, 'source_data_sha256': forecast.data.hashes,
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
    from experiments.exp008.blend_mode_budget_hold1_audit import audit
    data, forecast = Data(), BlendForecasts()
    prepare(forecast)
    save(OUT/'runtime_source_consistency_before.json', source_consistency())
    parts, audits, rows = [], [], []
    soc, mode, age = INITIAL_SOC, 1, 1
    for day in range(31, 365):
        issued = forecast.get(day, scenario='2')
        issued['audit']['load_method'] = PRIMARY
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
            save(OUT/'first3_planning_hashes.json', {str(day): digest(OUT/f'planning_day{day}.npz') for day in range(31, 34)})
            print('FIRST3_FEASIBILITY_PASSED; CONTINUE_FIXED334', flush=True)
    verification = verify_npz(OUT/'dispatch.npz', expected_days=334, audit_path=OUT/'planning_audit.json')
    assert verification['passed'], verification['errors']
    save(OUT/'summary.json', {'completed_days': 334, 'total_cost': float(arrays['fees'].sum()),
        'battery': battery_metrics(arrays), 'verification': verification, 'last_planned_mode': mode,
        'last_planned_run_slots': age, 'next_day_mode_hold_remaining': max(0, 1-age),
        'nonanticipative_recourse_certificate': False, 'same_forecast_full_year': True})
    checked = audit(expected_days=334)
    save(OUT/'runtime_source_consistency_after.json', source_consistency())
    return checked


if __name__ == '__main__':
    run()
