"""One fixed annual HGB experiment with current-state Q initialization and DP.

Every day's physical MIP and fixed-mask greedy Q refinement start from the
actual preceding SOC and real last nonidle direction. The actual controller
then uses the more flexible causal budget DP. This initializer is not claimed
to optimize the final feedback policy jointly; no block corrections are used.
"""
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

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.budget_feedback_diagnostic import check_flow
from experiments.exp008.closed_loop import optimize
from experiments.exp008.forecast_absolute_hgb import OUT as FORECAST_OUT, PRIMARY
from experiments.exp008.mode_budget_hold1_physical import plan
from experiments.exp008.verify import INITIAL_SOC, battery_metrics, verify_npz
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT / 'data/results/exp008/budget_feedback/physical334_hgb_cap8_grid200'
BASE = ROOT / 'data/results/exp008/mode_budget_hold1_physical/direct_hgb_memory_hold1_cap8_kappa0_334days'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def local_source_files():
    result = set()
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix == '.py' and '.venv' not in path.parts:
                result.add(path)
    return sorted(result)


def prepare(forecast, execution_data):
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = dict(candidate_count=1, evaluation_days=334, start_day=31, stop_exclusive=365,
        first3_gate='feasible physical MIP; independently verified physical actual flows and prefix causality',
        first3_cost_is_not_a_gate=True, no_stopping_from_partial_cost_or_count=True,
        same_unmodified_run_continues_from_first3=True, forecast=PRIMARY,
        initial_SOC=INITIAL_SOC, initial_previous_real_nonidle_direction=1,
        daily_real_switch_budget=8, midnight_reverse_counts=True, idle_preserves_last_direction=True,
        budget_resets_daily_while_SOC_and_last_real_nonidle_direction_carry=True,
        initial_MIP_scenarios=3, initial_scenario_indices='equally spaced in all28 chronological paths',
        initial_MIP_seconds=5., initial_MIP_gap=.002, initial_MIP_hold_slots=1,
        initial_MIP_daily_planned_mode_budget=8, initial_MIP_switch_penalty=0.,
        MIP_previous_mode='actual last nonidle direction', MIP_previous_run_slots=1,
        MIP_previous_age_has_no_lock_effect_under_hold1=True,
        Q_refinement='same full28 exact fixed-mask greedy objective, L-BFGS-B maxiter120',
        Q_initialization_uses_own_real_SOC_each_day=True, archive_Q_never_reused=True,
        fixed_mask_assumption_is_an_approximate_initializer_for_more_flexible_actual_DP=True,
        no_exact_joint_purchase_feedback_optimality_claim=True,
        Q_coordinate_correction=False, selection_between_MIP_and_refined_Q=False,
        DP_SOC_grid_kwh=200., DP_previous_error_bins=3, DP_conditional_tail_points=3,
        DP_transition_shrinkage_pseudocount=12., historical_completed_error_days=28,
        wear=.002, variation=0., charge_deadband=0.,
        terminal='MIP known minimum price/ETA; greedy and DP .45; all0 on day364',
        actual_bill='p*original_Q + 5*p*emergency only', actual_current_observation_before_action=True,
        no_simultaneous_charge_discharge=True, no_emergency_charging=True,
        physical_battery_parameters_unchanged=True, development_year_not_independent_test=True,
        conditional_five_day_trial='budget_feedback/fixed5_hgb_cap8_grid200_blocks8_step50',
        run_selected_after_development_pilot=True, no_annual_improvement_assumed=True)
    save(OUT / 'evaluation_protocol.json', protocol)
    files = local_source_files()
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = OUT / 'source_archive' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copy2(ROOT / name, OUT / name)
    artifacts = sorted([p for p in FORECAST_OUT.iterdir() if p.is_file()] + list((FORECAST_OUT / 'models').glob('*.joblib')))
    forecast_hashes = {}
    for path in artifacts:
        relative = path.relative_to(FORECAST_OUT)
        target = OUT / 'forecast_archive' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        forecast_hashes[str(relative)] = digest(target)
    np.savez_compressed(OUT / 'issued_forecasts.npz', origins=forecast.store.origins, values=forecast.store.values)
    save(OUT / 'complete_provenance.json', dict(source_sha256=hashes,
        runtime_local_import_files=[str(p.relative_to(ROOT)) for p in files],
        source_data_sha256=forecast.data.hashes, execution_source_data_sha256=execution_data.hashes,
        forecast_artifact_sha256=forecast_hashes,
        issued_forecast_archive_sha256=digest(OUT / 'issued_forecasts.npz'),
        forecast_values_sha256=array_digest(forecast.store.values), forecast_origins_sha256=array_digest(forecast.store.origins),
        original_hold1_dispatch_sha256=digest(BASE / 'dispatch.npz'),
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, primary=PRIMARY))
    save(OUT / 'functional_DP_checks.json', policy.checks())


def checkpoint(parts, audits, rows):
    arrays = {key: np.stack([p[key] for p in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 31 + len(parts))
    np.savez_compressed(OUT / 'dispatch_2.npz', **arrays)
    save(OUT / 'planning_audit.json', audits)
    pd.DataFrame(rows).to_csv(OUT / 'daily.csv', index=False)
    return arrays


def run():
    # Import before source closure is frozen; the auditor does not run a MIP.
    from experiments.exp008.budget_feedback_physical_audit import audit
    data = Data()
    # The shared forecast adapter retains a third observed-price channel for
    # its general Q4 interface. Q2 execution uses only the permitted two
    # power channels; unused price-error output is never passed to the DP.
    forecast = AbsoluteHGBForecasts()
    np.testing.assert_array_equal(forecast.data.actual[:, :2], data.actual)
    np.testing.assert_array_equal(forecast.data.fixed_price, data.fixed_price)
    prepare(forecast, data)
    soc, mode = INITIAL_SOC, 1
    parts, audits, rows = [], [], []
    for day in range(31, 365):
        issued = forecast.get(day, scenario='2')
        issued['audit']['load_method'] = 'direct_absolute_HGB_Ridge28_memory_half'
        errors = forecast.net_error_paths(day, '2', limit=28)
        net = (issued['load_kw'] - issued['pv_kw']) / 6
        paths = net[None, :] + errors['errors_kwh']
        assert paths.shape == (28, 144)
        selected = np.linspace(0, 27, 3).astype(int)
        try:
            initialized = plan(paths[selected], data.fixed_price, soc, mode, 1,
                seconds=5., gap=.002, final=day == 364)
        except (RuntimeError, AssertionError) as exc:
            if parts:
                checkpoint(parts, audits, rows)
            save(OUT / 'failure.json', dict(day=day, completed_days=len(parts), error=str(exc),
                no_added_time_or_parameter_change=True, annual_complete=False))
            raise
        refined = optimize(initialized['purchase'], paths, data.fixed_price, soc,
            charge_mask=initialized['allowed_charge'], throughput=.002, variation=0.,
            terminal=0. if day == 364 else .45, deadband=0., maxiter=120)
        q = refined['purchase']
        model = policy.fit_error_model(net, errors['errors_kwh'], points=3,
            history_origins=errors['origins'], cutoff=day * 144)
        grid, values, dp_meta = policy.value_functions(q, model, data.fixed_price,
            grid_kwh=200., wear=.002, terminal=0. if day == 364 else .45, budget=8)
        np.savez_compressed(OUT / f'planning_day{day}.npz',
            **{k: v for k, v in initialized.items() if k != 'metadata'},
            refined_Q=q, all_net_paths=paths, historical_errors=errors['errors_kwh'],
            history_origins=errors['origins'], forecast_net=net, selected_scenario_indices=selected,
            initial_soc=soc, previous_real_mode=mode, previous_MIP_run_slots=1,
            price=data.fixed_price, DP_grid=grid)
        np.savez_compressed(OUT / f'feedback_model_day{day}.npz',
            **{k: v for k, v in model.items() if k != 'metadata'})
        locked = dict(planning_sha256=digest(OUT / f'planning_day{day}.npz'),
            model_sha256=digest(OUT / f'feedback_model_day{day}.npz'),
            value_table_sha256=array_digest(values), purchase_sha256=array_digest(q))
        # This day's actual net-demand row is first accessed after Q and its
        # policy table have been formed and their file/array hashes recorded.
        actual = data.actual[day * 144:(day + 1) * 144].copy()
        observed_net = ((actual[:, 0] - actual[:, 1]) / 6)[None, :]
        flow = policy.execute_paths(q, observed_net, data.fixed_price, soc, grid, values, model,
            wear=.002, initial_mode=mode, budget=8)
        error = check_flow(q, observed_net, data.fixed_price, flow, soc, mode)
        detail = {k: v[0].copy() for k, v in flow.items()}
        detail.update(original=q.copy(), final=q.copy(), actual=actual, price=data.fixed_price.copy(),
            initializer_allowed_charge=initialized['allowed_charge'].copy())
        next_mode = int(flow['modes'][0, -1])
        used = 8 - int(flow['remaining_budget'][0, -1])
        record = dict(day=day, forecast=issued['audit'], history=errors['audit'],
            mip=initialized['metadata'], greedy_refinement=refined['metadata'], feedback_DP=dp_meta,
            feedback_model=model['metadata'], original_previous_real_mode=mode, final_real_mode=next_mode,
            initial_real_soc=soc, final_real_soc=float(flow['states'][0, -1]), actual_switches_with_boundary=used,
            maximum_actual_physical_error=error, locked_before_current_actual=locked,
            no_actual_future_policy_actions=True)
        audits.append(record)
        parts.append(detail)
        rows.append(dict(day=day, date=str((pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).date()),
            total_cost=float(detail['fees'].sum()), planned_cost=float(detail['fees'][:, 0].sum()),
            emergency_cost=float(detail['fees'][:, 3].sum()), initial_soc=soc,
            final_soc=record['final_real_soc'], initial_real_mode=mode, final_real_mode=next_mode,
            actual_switches_with_boundary=used, mip_gap=initialized['metadata']['mip_gap'],
            mip_seconds=initialized['metadata']['seconds'], greedy_success=refined['metadata']['success'],
            planned_switches=initialized['metadata']['planned_changes_including_boundary']))
        soc, mode = record['final_real_soc'], next_mode
        print('BUDGET_DAY', day, 'fee', rows[-1]['total_cost'], 'switches', used,
            'SOC', soc, 'MIP_gap', rows[-1]['mip_gap'], flush=True)
        if len(parts) == 3 or len(parts) % 10 == 0 or len(parts) == 334:
            arrays = checkpoint(parts, audits, rows)
        if len(parts) == 3:
            checked = audit(3)
            gate = dict(passed=checked['passed'], completed_days=3,
                cost_and_count_not_used=True, MIP_gap_not_a_feasibility_gate=True,
                independent_audit_sha256=digest(OUT / 'independent_audit_first3.json'))
            save(OUT / 'feasibility_gate.json', gate)
            np.savez_compressed(OUT / 'first3_dispatch.npz', **arrays)
            save(OUT / 'first3_planning_hashes.json', {str(d): digest(OUT / f'planning_day{d}.npz') for d in range(31, 34)})
            if not gate['passed']:
                raise AssertionError('Predeclared first3 physical and causal gate failed')
            print('FIRST3_BUDGET_PHYSICS_CAUSALITY_PASSED_CONTINUE334', flush=True)
    verification = verify_npz(OUT / 'dispatch_2.npz', expected_days=334, audit_path=OUT / 'planning_audit.json')
    assert verification['passed'], verification['errors']
    save(OUT / 'summary.json', dict(completed_days=334, total_cost=float(arrays['fees'].sum()),
        planned_cost=float(arrays['fees'][:, :, 0].sum()), emergency_cost=float(arrays['fees'][:, :, 3].sum()),
        battery=battery_metrics(arrays), verification=verification, own_continuous_SOC_and_real_direction=True,
        original_Q_algorithm_preserved_but_recomputed_from_own_state=True,
        q_feedback_joint_optimality_not_claimed=True, maximum_daily_actual_switches=max(r['actual_switches_with_boundary'] for r in rows),
        annual_actual_switch_upper_bound=2672))
    return audit(334)


if __name__ == '__main__':
    run()
