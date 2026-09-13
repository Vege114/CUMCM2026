"""Independent scalar Bellman/accounting checks for the five-day DP pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.budget_feedback_diagnostic import BASE, DAYS, OUT, check_flow, load, save
from experiments.exp008.full28_mode_diagnostic_audit import sha
from experiments.exp008.verify import verify_arrays

ROOT = Path(__file__).resolve().parents[2]


def scalar_bellman(q, price, grid, values, model, t, previous_bin, mode_index, remaining, soc_index, wear=.002):
    """Enumerate scalar actions without invoking the vectorized DP builder."""
    current = grid[soc_index]
    previous_mode = (-1, 1)[mode_index]
    total = 0.
    for observed_bin in range(3):
        conditional = 0.
        for emission in range(3):
            demand = model['forecast_net_kwh'][t] + model['conditional_error_support_kwh'][t, observed_bin, emission]
            balance = q[t] - demand
            direction = 1 if balance >= 0 else -1
            switch = int(direction != previous_mode)
            continuation = values[t + 1, observed_bin]
            best = float(continuation[mode_index, remaining, soc_index] + 5 * price[t] * max(-balance, 0.))
            if remaining >= switch:
                if direction == 1:
                    available = min(balance, policy.LIMIT, (policy.HIGH - current) / policy.ETA)
                    endpoint = current + policy.ETA * available
                    choices = [s for s in grid if current < s <= endpoint]
                    if available > 1e-8:
                        choices.append(endpoint)
                else:
                    available = min(-balance, policy.LIMIT, (current - policy.LOW) * policy.ETA)
                    endpoint = current - available / policy.ETA
                    choices = [s for s in grid if endpoint <= s < current]
                    if available > 1e-8:
                        choices.append(endpoint)
                for following in choices:
                    charge = max(0., (following - current) / policy.ETA)
                    discharge = max(0., (current - following) * policy.ETA)
                    stage = 5 * price[t] * max(-balance - discharge, 0.) + wear * (charge + discharge)
                    value = stage + np.interp(following, grid, continuation[int(direction > 0), remaining - switch])
                    best = min(best, float(value))
            conditional += model['conditional_emission_weights'][t, observed_bin, emission] * best
        total += model['transition'][t, previous_bin, observed_bin] * conditional
    return total


def run():
    protected = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    assert sha(OUT / 'protocol.json') == manifest['protocol_sha256']
    for relative, digest in manifest['input_files'].items():
        assert sha(BASE / relative) == digest
    for relative, digest in manifest['source_files'].items():
        assert sha(OUT / 'source_archive' / relative) == digest
        assert sha(ROOT / relative) == digest
    locks = json.loads((OUT / 'all_choices_locked_before_actual.json').read_text())
    assert not locks['actual_CSV_loaded']
    source = load(BASE / 'dispatch.npz')
    actual = np.stack([pd.read_csv(ROOT / 'data/raw' / name).iloc[:, 1:].to_numpy(float)
        for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    records, candidates_checked, bellman_count, max_bellman_error = [], 0, 0, 0.
    for day in DAYS:
        p = load(OUT / f'day{day}_inputs.npz')
        selected = json.loads((OUT / f'day{day}_selection.json').read_text())
        model = policy.fit_error_model(p['forecast_net'], p['historical_errors'], points=3,
            history_origins=p['history_origins'], cutoff=day * 144)
        frozen_model = load(OUT / f'day{day}_model.npz')
        for key, value in model.items():
            if key != 'metadata':
                np.testing.assert_array_equal(value, frozen_model[key])
        assert int(p['history_origins'][-1]) + 144 <= day * 144
        assert int(p['history_origins'][0]) == (day - 28) * 144
        previous_power = 6 * (source['charge'] - source['discharge']).ravel()[:(day - 31) * 144]
        nonidle = previous_power[np.abs(previous_power) > 1e-6]
        mode = int(np.sign(nonidle[-1])) if len(nonidle) else 1
        power = float(previous_power[-1]) if len(previous_power) else 172.75999999999976
        assert mode == int(p['initial_real_mode']) and power == float(p['initial_power_kw'])
        np.testing.assert_array_equal(p['original_Q'], source['original'][day - 31])
        np.testing.assert_array_equal(p['initial_soc'], source['states'][day - 31, 0])
        terminal = 0. if day == 364 else .45
        choices = load(OUT / f'day{day}_all_candidate_Q.npz')
        recomputed, cache = [], {}
        for index in range(17):
            q = choices['q'][index]
            np.testing.assert_array_equal(q, np.maximum(0., p['original_Q'] + np.repeat(choices['theta'][index], 18)))
            grid, values, _ = policy.value_functions(q, model, p['price'], terminal=terminal)
            flow = policy.execute_paths(q, p['all_net_paths'], p['price'], float(p['initial_soc']), grid, values, model,
                initial_mode=mode)
            check_flow(q, p['all_net_paths'], p['price'], flow, float(p['initial_soc']), mode)
            # Sum components here rather than calling historical_objective.
            bill = q @ p['price'] + np.mean(np.sum(5 * flow['emergency'] * p['price']
                + .002 * (flow['charge'] + flow['discharge']), axis=1))
            exact = float(bill - terminal * np.mean(flow['states'][:, -1] - policy.LOW))
            approx = float(q @ p['price'] + np.interp(float(p['initial_soc']), grid,
                values[0, model['initial_error_state'], int(mode > 0), 8]))
            metadata = selected['candidates'][index]
            np.testing.assert_allclose(exact, metadata['historical_policy_objective'], atol=1e-6, rtol=0.)
            np.testing.assert_allclose(approx, metadata['DP_initial_expected_objective'], atol=1e-6, rtol=0.)
            recomputed.append(exact)
            candidates_checked += 1
            if index in (0, selected['selected_index']):
                cache[index] = (grid, values, flow)
        # Reconstruct sequential one-sweep choice, rather than arbitrary best-of-17.
        best_index, theta = 0, np.zeros(8)
        for block in range(8):
            initial_theta, next_theta = theta.copy(), theta.copy()
            for sign, index in zip((-1, 1), (2 * block + 1, 2 * block + 2)):
                expected_theta = initial_theta.copy()
                expected_theta[block] += 50 * sign
                np.testing.assert_array_equal(expected_theta, choices['theta'][index])
                if recomputed[index] < recomputed[best_index] - 1e-7:
                    best_index, next_theta = index, expected_theta
            theta = next_theta
        assert best_index == selected['selected_index']
        np.testing.assert_array_equal(theta, selected['selected_theta_kwh'])
        assert sha(OUT / f'day{day}_selected_policy.npz') == locks['choices'][str(day)] == selected['selected_policy_sha256']
        for label, index in (('initial', 0), ('selected', best_index)):
            grid, values, history = cache[index]
            saved_policy = load(OUT / f'day{day}_{label}_policy.npz')
            saved_history = load(OUT / f'day{day}_{label}_history.npz')
            for key, value in dict(q=choices['q'][index], grid=grid, values=values).items():
                np.testing.assert_array_equal(saved_policy[key], value)
            for key, value in history.items():
                np.testing.assert_array_equal(saved_history[key], value)
            for t in (0, 36, 72, 108, 143):
                for prior_bin in range(3):
                    for mode_index in range(2):
                        for remaining in (0, 4, 8):
                            for soc_index in (0, 24, 48):
                                scalar = scalar_bellman(choices['q'][index], p['price'], grid, values, model,
                                    t, prior_bin, mode_index, remaining, soc_index)
                                delta = abs(scalar - values[t, prior_bin, mode_index, remaining, soc_index])
                                max_bellman_error = max(max_bellman_error, float(delta))
                                assert delta < 1e-6
                                bellman_count += 1
            net = ((actual[day, :, 0] - actual[day, :, 1]) / 6)[None, :]
            flow = policy.execute_paths(choices['q'][index], net, p['price'], float(p['initial_soc']),
                grid, values, model, initial_mode=mode)
            archived = load(OUT / f'day{day}_{label}_actual_replay.npz')
            for key, value in flow.items():
                np.testing.assert_array_equal(value, archived[key])
            detail = dict(original=choices['q'][index][None, :], final=choices['q'][index][None, :],
                actual=actual[day:day + 1], price=p['price'][None, :], **flow)
            audit = verify_arrays(detail, expected_days=1, start_day=day, initial_soc=float(p['initial_soc']),
                initial_mode=mode, initial_power_kw=power, source_actual=actual[day:day + 1], source_price=p['price'][None, :])
            assert audit['passed'], audit
            for stop in (1, 36, 108):
                changed = net.copy()
                changed[:, stop:] += np.linspace(1e4, 7e4, 144 - stop)
                alt = policy.execute_paths(choices['q'][index], changed, p['price'], float(p['initial_soc']),
                    grid, values, model, initial_mode=mode)
                for key in ('charge', 'discharge', 'emergency', 'surplus'):
                    np.testing.assert_array_equal(flow[key][:, :stop], alt[key][:, :stop])
                for key in ('states', 'modes', 'remaining_budget'):
                    np.testing.assert_array_equal(flow[key][:, :stop + 1], alt[key][:, :stop + 1])
        records.append(dict(day=day, model_reconstructed_exactly=True, prior_real_mode_and_SOC_passed=True,
            seventeen_Q_objectives_recomputed=True, historical_coordinate_selection_exact=True,
            initial_and_selected_policy_tables_and_history_and_actual_replayed_exactly=True,
            independent_verify_arrays_passed=True, current_actual_prefix_checks=6))
    assert protected == {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    result = dict(passed=True, candidate_DP_and_full28_history_replays=candidates_checked,
        scalar_Bellman_checks=bellman_count, maximum_Bellman_error=max_bellman_error,
        fixed_five_day_source_checks=records, all_input_and_source_hashes_unchanged=True,
        outputs_preserved=True, annual_goal_not_evaluated=True,
        provenance_limit='Model fit causality inherits the already audited issued-HGB history archive; this audit reconstructs its frozen inputs and cutoff.')
    save(OUT / 'independent_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
