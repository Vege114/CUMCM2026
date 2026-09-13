"""Audit the continuous annual budget-DP candidate without rerunning any MIP."""
from __future__ import annotations

import json

import numpy as np
from experiments.exp008 import budget_feedback as policy
from experiments.exp008.run_blend_mode_budget_hold1 import BlendForecasts
from experiments.exp008.forecast_hgb_extra_trees_half import OUT as FORECAST_OUT, PRIMARY
from experiments.exp008.budget_feedback_diagnostic import check_flow, load
from experiments.exp008.closed_loop import optimize
from experiments.exp008.run_blend_budget_feedback_physical import (BASE, OUT, ROOT, array_digest,
    digest, local_source_files, save)
from experiments.exp008.verify import INITIAL_SOC, battery_metrics, verify_npz
from experiments.problem2.exp003.data import Data


def forecast_future_check(day, expected_forecast, expected_errors):
    altered = BlendForecasts()
    altered.data.actual[day * 144:] += np.array([70000., 30000., 1000.])
    altered.store.values[altered.store.origins > day * 144] += np.array([60000., 20000.])
    issued = altered.get(day, scenario='2')
    error = altered.net_error_paths(day, '2', limit=28)
    np.testing.assert_array_equal((issued['load_kw'] - issued['pv_kw']) / 6, expected_forecast)
    np.testing.assert_array_equal(error['errors_kwh'], expected_errors)
    return dict(day=day, altered_current_and_future_actual=True,
        altered_unreleased_future_forecast_channels_asymmetrically=True,
        current_issue_and_all28_completed_errors_unchanged=True)


def audit(expected_days=334):
    arrays = load(OUT / 'dispatch_2.npz')
    assert arrays['original'].shape == (expected_days, 144)
    assert np.array_equal(arrays['days'], np.arange(31, 31 + expected_days))
    records = json.loads((OUT / 'planning_audit.json').read_text())
    provenance = json.loads((OUT / 'complete_provenance.json').read_text())
    for name, value in provenance['source_sha256'].items():
        assert digest(ROOT / name) == value, name
        assert digest(OUT / 'source_archive' / name) == value, name
    current_imports = {str(p.relative_to(ROOT)) for p in local_source_files()}
    assert current_imports <= set(provenance['runtime_local_import_files']), sorted(current_imports - set(provenance['runtime_local_import_files']))
    for name, value in provenance['forecast_artifact_sha256'].items():
        assert digest(OUT / 'forecast_archive' / name) == value
    for path, value in provenance['forecast_original_sha256'].items():
        assert digest(path) == value
    assert json.loads((OUT/'forecast_archive'/FORECAST_OUT.name/'independent_verification.json').read_text())['passed']
    assert json.loads((OUT/'forecast_archive'/FORECAST_OUT.name/'causality_verification.json').read_text())['passed']
    assert digest(OUT / 'issued_forecasts.npz') == provenance['issued_forecast_archive_sha256']
    assert digest(BASE / 'dispatch.npz') == provenance['original_hold1_dispatch_sha256']
    data = Data()
    forecast = BlendForecasts()
    np.testing.assert_array_equal(forecast.data.actual[:, :2], data.actual)
    np.testing.assert_array_equal(forecast.data.fixed_price, data.fixed_price)
    assert data.hashes == provenance['execution_source_data_sha256']
    assert forecast.data.hashes == provenance['source_data_sha256']
    assert array_digest(forecast.store.values) == provenance['forecast_values_sha256']
    assert array_digest(forecast.store.origins) == provenance['forecast_origins_sha256']
    verification = verify_npz(OUT / 'dispatch_2.npz', expected_days=expected_days,
        audit_path=OUT / 'planning_audit.json')
    assert verification['passed'], verification['errors']
    previous_mode, previous_soc = 1, INITIAL_SOC
    worst, replayed, mutations, mip_checks = 0., 0, 0, 0
    sampled_reproductions, sampled_future_checks = [], []
    if expected_days == 3:
        sample_days = (31, 32, 33)
    else:
        sample_days = (31, 90, 151, 243, 364)
    for i, record in enumerate(records):
        day = 31 + i
        p = load(OUT / f'planning_day{day}.npz')
        frozen_model = load(OUT / f'feedback_model_day{day}.npz')
        locked = record['locked_before_current_actual']
        assert digest(OUT / f'planning_day{day}.npz') == locked['planning_sha256']
        assert digest(OUT / f'feedback_model_day{day}.npz') == locked['model_sha256']
        assert array_digest(arrays['original'][i]) == locked['purchase_sha256']
        assert p['previous_real_mode'] == previous_mode == arrays['modes'][i, 0]
        assert p['initial_soc'] == previous_soc == arrays['states'][i, 0]
        assert p['previous_MIP_run_slots'] == 1
        assert record['mip']['previous_planned_mode'] == previous_mode
        assert record['mip']['previous_planned_run_slots'] == 1
        assert record['mip']['hold_slots'] == 1
        assert record['mip']['scenario_count'] == 3
        assert record['mip']['switching_cost'] == 0
        assert record['mip']['daily_planned_switch_cap'] == 8
        np.testing.assert_array_equal(p['selected_scenario_indices'], np.array([0, 13, 27]))
        np.testing.assert_array_equal(p['refined_Q'], arrays['original'][i])
        c, d, e, w, s = [p['scenario_' + key] for key in ('charge', 'discharge', 'emergency', 'surplus', 'states')]
        selected_paths = p['all_net_paths'][p['selected_scenario_indices']]
        balance_error = float(np.abs(p['purchase'][None, :] + d + e - c - w - selected_paths).max())
        soc_error = float(np.abs(np.diff(s, axis=1) - policy.ETA * c + d / policy.ETA).max())
        worst = max(worst, balance_error, soc_error)
        assert max(balance_error, soc_error) < 1e-6
        assert min(c.min(), d.min(), e.min(), w.min()) >= -1e-6
        assert s.min() >= policy.LOW - 1e-6 and s.max() <= policy.HIGH + 1e-6
        assert max(c.max(), d.max()) <= policy.LIMIT + 1e-6
        assert np.all(s[:, 0] == previous_soc)
        assert not np.any((c > 1e-6) & ((d > 1e-6) | (e > 1e-6)))
        assert np.all(c[:, ~p['allowed_charge']] <= 1e-6) and np.all(d[:, p['allowed_charge']] <= 1e-6)
        flips = np.r_[p['allowed_charge'][0] != (previous_mode > 0), p['allowed_charge'][1:] != p['allowed_charge'][:-1]]
        np.testing.assert_array_equal(flips, p['planned_flips'])
        np.testing.assert_allclose(flips, p['solved_flip_values'], atol=1e-6, rtol=0.)
        assert int(flips.sum()) <= 8
        mip_checks += 1
        issued = forecast.get(day, scenario='2')
        assert record['forecast']['selected_model_id'] == PRIMARY
        assert record['forecast']['load_method'] == PRIMARY
        assert record['history']['model_id'] == PRIMARY
        errors = forecast.net_error_paths(day, '2', limit=28)
        net = (issued['load_kw'] - issued['pv_kw']) / 6
        np.testing.assert_array_equal(net, p['forecast_net'])
        np.testing.assert_array_equal(errors['errors_kwh'], p['historical_errors'])
        np.testing.assert_array_equal(errors['origins'], p['history_origins'])
        np.testing.assert_array_equal(net[None, :] + errors['errors_kwh'], p['all_net_paths'])
        model = policy.fit_error_model(net, errors['errors_kwh'], points=3,
            history_origins=errors['origins'], cutoff=day * 144)
        for name, value in model.items():
            if name != 'metadata':
                np.testing.assert_array_equal(value, frozen_model[name])
        terminal = 0. if day == 364 else .45
        grid, values, _ = policy.value_functions(p['refined_Q'], model, data.fixed_price,
            grid_kwh=200., wear=.002, terminal=terminal, budget=8)
        assert array_digest(values) == locked['value_table_sha256']
        np.testing.assert_array_equal(grid, p['DP_grid'])
        actual = data.actual[day * 144:(day + 1) * 144]
        actual_net = ((actual[:, 0] - actual[:, 1]) / 6)[None, :]
        replay = policy.execute_paths(p['refined_Q'], actual_net, data.fixed_price, previous_soc,
            grid, values, model, initial_mode=previous_mode)
        worst = max(worst, check_flow(p['refined_Q'], actual_net, data.fixed_price, replay, previous_soc, previous_mode))
        for key, value in replay.items():
            np.testing.assert_array_equal(value[0], arrays[key][i])
        for stop in (1, 36, 108):
            changed = actual_net.copy()
            changed[:, stop:] += np.linspace(10000., 70000., 144 - stop)
            alternative = policy.execute_paths(p['refined_Q'], changed, data.fixed_price, previous_soc,
                grid, values, model, initial_mode=previous_mode)
            for key in ('charge', 'discharge', 'emergency', 'surplus'):
                np.testing.assert_array_equal(replay[key][:, :stop], alternative[key][:, :stop])
            for key in ('states', 'modes', 'remaining_budget'):
                np.testing.assert_array_equal(replay[key][:, :stop + 1], alternative[key][:, :stop + 1])
            mutations += 1
        if day in sample_days:
            refined = optimize(p['purchase'], p['all_net_paths'], data.fixed_price, previous_soc,
                charge_mask=p['allowed_charge'], throughput=.002, variation=0., terminal=terminal,
                deadband=0., maxiter=120)
            np.testing.assert_array_equal(refined['purchase'], p['refined_Q'])
            sampled_reproductions.append(day)
            sampled_future_checks.append(forecast_future_check(day, net, errors['errors_kwh']))
        replayed += 1
        previous_mode, previous_soc = int(replay['modes'][0, -1]), float(replay['states'][0, -1])
    assert len(records) == expected_days
    used = 8 - arrays['remaining_budget'][:, -1]
    assert np.all(used <= 8) and np.all(used >= 0)
    nonidle = np.sign((arrays['charge'] - arrays['discharge']).ravel())
    nonidle = nonidle[np.abs((arrays['charge'] - arrays['discharge']).ravel()) > 1e-6]
    actual_reversals = int(np.sum(np.diff(np.r_[1, nonidle]) != 0))
    assert actual_reversals <= int(used.sum()) <= 8 * expected_days
    # If tiny numerical actions consume budget, the reported >1e-6 counts may
    # be smaller. They can never invalidate the hard upper bound.
    result = dict(passed=True, completed_days=expected_days, verification=verification,
        source_closure_files=len(provenance['source_sha256']), runtime_import_closure_unchanged=True,
        all_original_forecast_models_and_input_hashes_verified=True,
        all_days_reconstructed_from_same_issued_forecasts_and_all28_completed_errors=True,
        every_MIP_initialized_with_own_real_SOC_and_last_real_nonidle_direction=True,
        MIP_physical_checks=mip_checks, every_DP_model_and_value_table_exact=True,
        current_actual_replay_days=replayed, future_actual_prefix_mutations=mutations,
        sampled_refine120_Q_exact_reproduction_days=sampled_reproductions,
        future_forecast_and_actual_mutations=sampled_future_checks,
        max_physical_error_kwh=worst, total_budget_consumed=int(used.sum()),
        real_reversals_including_initial_boundary=actual_reversals,
        maximum_day_budget_consumed=int(used.max()),
        fixed_mask_MIP_and_greedy_remain_approximate_Q_initializers=True,
        no_joint_global_optimality_or_battery_lifetime_claim=True)
    if expected_days == 334:
        baseline = load(BASE / 'dispatch.npz')
        np.testing.assert_array_equal(arrays['actual'], baseline['actual'])
        np.testing.assert_array_equal(arrays['price'], baseline['price'])
        new, old = battery_metrics(arrays), battery_metrics(baseline)
        keys = ('direction_reversals', 'direction_reversals_including_warmup_boundary', 'charge_starts',
            'discharge_starts', 'throughput_kwh', 'equivalent_full_cycles', 'active_slots',
            'power_ramp_total_kw', 'final_soc')
        result['comparison_to_hold1'] = dict(total_cost_delta=float(arrays['fees'].sum() - baseline['fees'].sum()),
            planned_cost_delta=float(arrays['fees'][:, :, 0].sum() - baseline['fees'][:, :, 0].sum()),
            emergency_cost_delta=float(arrays['fees'][:, :, 3].sum() - baseline['fees'][:, :, 3].sum()),
            battery_deltas={key: new[key] - old[key] for key in keys},
            episode_delta=(new['charge_starts'] + new['discharge_starts'] - old['charge_starts'] - old['discharge_starts']))
        first = load(OUT / 'first3_dispatch.npz')
        for key, value in first.items():
            np.testing.assert_array_equal(value, arrays[key][:3])
        result['first3_prefix_preserved_exactly'] = True
        for day, digest_value in json.loads((OUT / 'first3_planning_hashes.json').read_text()).items():
            assert digest(OUT / f'planning_day{day}.npz') == digest_value
    destination = OUT / ('independent_audit_first3.json' if expected_days == 3 else 'independent_audit.json')
    save(destination, result)
    print('INDEPENDENT_BUDGET_AUDIT_PASSED', expected_days, 'physical_error', worst, flush=True)
    return result
