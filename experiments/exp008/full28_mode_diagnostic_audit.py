"""Read-only reconstruction of the completed five-day full28 mode diagnostic.

No optimizer is called. Existing solves, selections and trajectories are never
overwritten; only this audit and a matched terminal-sensitivity table are added.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008/full28_mode_diagnostic/hold1_cap8_fixed5_60seconds'
BASE = ROOT / 'data/results/exp008/mode_budget_hold1_physical/direct_hgb_memory_hold1_cap8_kappa0_334days'
ETA, LOW, HIGH, LIMIT = np.sqrt(.9), 1200., 10800., 5000 / 6
DAYS = (31, 90, 151, 243, 364)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def reconstruct(q, net, mask, soc):
    k, n = net.shape
    states = np.full((k, n + 1), soc)
    c, d, e, w = [np.zeros((k, n)) for _ in range(4)]
    for t in range(n):
        residual = q[t] - net[:, t]
        if mask[t]:
            c[:, t] = np.minimum.reduce([np.maximum(residual, 0.),
                np.full(k, LIMIT), np.maximum((HIGH - states[:, t]) / ETA, 0.)])
        else:
            d[:, t] = np.minimum.reduce([np.maximum(-residual, 0.),
                np.full(k, LIMIT), np.maximum((states[:, t] - LOW) * ETA, 0.)])
        e[:, t] = np.maximum(-residual - d[:, t], 0.)
        w[:, t] = np.maximum(residual - c[:, t], 0.)
        states[:, t + 1] = states[:, t] + ETA * c[:, t] - d[:, t] / ETA
    return dict(charge=c, discharge=d, emergency=e, surplus=w, states=states)


def check_physics(q, net, mask, flow):
    c, d, e, w, s = [flow[k] for k in ('charge', 'discharge', 'emergency', 'surplus', 'states')]
    error = max(float(np.abs(q[None, :] + d + e - c - w - net).max()),
                float(np.abs(np.diff(s, axis=1) - ETA * c + d / ETA).max()))
    assert error < 1e-6
    assert min(c.min(), d.min(), e.min(), w.min()) >= -1e-6
    assert max(c.max(), d.max()) <= LIMIT + 1e-6
    assert s.min() >= LOW - 1e-6 and s.max() <= HIGH + 1e-6
    assert not np.any((c > 1e-6) & ((d > 1e-6) | (e > 1e-6)))
    assert np.all(c[:, ~mask] <= 1e-6) and np.all(d[:, mask] <= 1e-6)
    assert np.all(d <= np.maximum(net - q, 0.) + 1e-6)
    return error


def run():
    protected = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, digest in protocol['protected_source_tree_sha256'].items():
        assert sha(BASE / name) == digest, name
    for name, digest in json.loads((OUT / 'source_sha256.json').read_text()).items():
        assert sha(OUT / 'source_archive' / name) == digest, name
    source = load(BASE / 'dispatch.npz')
    actual = np.stack([pd.read_csv(ROOT / 'data/raw' / name).iloc[:, 1:].to_numpy(float)
        for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    locks = json.loads((OUT / 'all_choices_locked_before_actual.json').read_text())
    assert not locks['current_actual_loaded']
    records, rows, worst, mutations = [], [], 0., 0
    for day in DAYS:
        selected = json.loads((OUT / f'day{day}_selection.json').read_text())
        p = load(BASE / f'planning_day{day}.npz')
        initial_soc = float(p['initial_soc'])
        np.testing.assert_allclose(initial_soc, source['states'][day - 31, 0], atol=0., rtol=0.)
        assert selected['full28_feasible'] and not selected['current_actual_used']
        selected_file = OUT / f'day{day}_{selected["selected"]}.npz'
        assert sha(selected_file) == selected['selected_sha256'] == locks['choices'][str(day)]['selected_sha256']
        mip = load(OUT / f'day{day}_full28_MIP.npz')
        flow = {k: mip['scenario_' + k] for k in ('charge', 'discharge', 'emergency', 'surplus', 'states')}
        worst = max(worst, check_physics(mip['purchase'], p['all_net_paths'], mip['allowed_charge'], flow))
        np.testing.assert_allclose(flow['states'][:, 0], initial_soc, atol=0., rtol=0.)
        expected_flips = np.r_[mip['allowed_charge'][0] != (int(p['previous_planned_mode']) > 0),
            mip['allowed_charge'][1:] != mip['allowed_charge'][:-1]]
        np.testing.assert_array_equal(mip['planned_flips'], expected_flips)
        np.testing.assert_allclose(mip['solved_flip_values'], expected_flips, atol=1e-6, rtol=0.)
        assert expected_flips.sum() <= 8
        histories = {}
        for label in ('source3_initial', 'source3_refined', 'full28_initial', 'full28_refined'):
            saved = load(OUT / f'day{day}_{label}.npz')
            np.testing.assert_array_equal(saved['paths'], p['all_net_paths'])
            np.testing.assert_array_equal(saved['price'], p['price'])
            np.testing.assert_allclose(saved['initial_soc'], initial_soc, atol=0., rtol=0.)
            flow = reconstruct(saved['q'], saved['paths'], saved['mask'], initial_soc)
            for key, values in flow.items():
                np.testing.assert_array_equal(values, saved[key])
            worst = max(worst, check_physics(saved['q'], saved['paths'], saved['mask'], flow))
            terminal = 0. if day == 364 else .45
            value = float(saved['q'] @ saved['price'] + np.mean(np.sum(
                5 * flow['emergency'] * saved['price'] + .002 * (flow['charge'] + flow['discharge']), axis=1))
                - terminal * np.mean(flow['states'][:, -1] - LOW))
            np.testing.assert_allclose(value, selected[label]['objective'], atol=1e-6, rtol=0.)
            histories[label] = value
            net = ((actual[day, :, 0] - actual[day, :, 1]) / 6)[None, :]
            replay = reconstruct(saved['q'], net, saved['mask'], initial_soc)
            archived = load(OUT / f'day{day}_{label}_actual_replay.npz')
            for key, values in replay.items():
                np.testing.assert_array_equal(values, archived[key])
                if label == 'source3_refined':
                    np.testing.assert_array_equal(values[0], source[key][day - 31])
            worst = max(worst, check_physics(saved['q'], net, saved['mask'], replay))
            fees = np.stack((saved['q'][None, :] * saved['price'], np.zeros((1, 144)),
                np.zeros((1, 144)), 5 * replay['emergency'] * saved['price']), axis=-1)
            np.testing.assert_array_equal(fees, archived['fees'])
            rows.append(dict(day=day, label=label, selected=label == selected['selected'],
                historical_objective=value, total_cost=float(fees.sum()),
                plan_cost=float(fees[:, :, 0].sum()), emergency_cost=float(fees[:, :, 3].sum()),
                terminal_soc=float(replay['states'][0, -1])))
            if label == selected['selected']:
                for stop in (1, 36, 108):
                    changed = net.copy()
                    changed[:, stop:] += np.linspace(1e4, 7e4, 144 - stop)
                    changed_replay = reconstruct(saved['q'], changed, saved['mask'], initial_soc)
                    for key in ('charge', 'discharge', 'emergency', 'surplus'):
                        np.testing.assert_array_equal(replay[key][:, :stop], changed_replay[key][:, :stop])
                    np.testing.assert_array_equal(replay['states'][:, :stop + 1], changed_replay['states'][:, :stop + 1])
                    mutations += 1
        expected_label = 'full28_refined' if histories['full28_refined'] <= histories['full28_initial'] else 'full28_initial'
        assert expected_label == selected['selected']
        records.append(dict(day=day, all_four_history_replays_exact=True,
            all_four_actual_replays_and_original_source_exact=True, selection_matches_historical_rule=True,
            MIP_gap=selected['full28_MIP_metadata']['mip_gap'], planned_switches=int(expected_flips.sum())))
    frame = pd.DataFrame(rows)
    pairs = []
    for day in DAYS:
        new = frame[(frame.day == day) & frame.selected].iloc[0]
        old = frame[(frame.day == day) & (frame.label == 'source3_refined')].iloc[0]
        delta = float(new.total_cost - old.total_cost)
        soc_delta = float(new.terminal_soc - old.terminal_soc)
        price = load(BASE / f'planning_day{day}.npz')['price']
        # Sensitivity only: stored energy has no cash refund in the actual fee.
        terminal_unit = 0. if day == 364 else .45
        pairs.append(dict(day=day, actual_cost_delta=delta, terminal_soc_delta_kwh=soc_delta,
            internal_terminal_adjusted_delta=delta-terminal_unit*soc_delta,
            maximum_single_future_slot_emergency_value_adjusted_delta=delta-5*float(price.max())*ETA*soc_delta,
            adjustment_is_sensitivity_not_realized_annual_saving=True))
    result = dict(passed=True, existing_solves_rerun=False, original_outputs_unchanged=True,
        original_source_tree_hashes_passed=True, six_recorded_source_hashes_passed=True,
        archived_source_list_is_not_a_complete_transitive_dependency_certificate=True,
        maximum_physical_error_kwh=worst, future_net_mutations=mutations,
        historical_selection_checked=True, actual_fee_reconstructed_from_raw_load_pv=True,
        prices_match_frozen_source=True, full365_or334_day_conclusion_supported=False,
        daily_records=records, terminal_sensitivity=pairs,
        limitations=['Five independently initialized days; changed ending SOC is not propagated.',
            'Execution prefix mutation proves only the fixed-plan execution rule, not optimality.',
            'Historical choice ordering additionally relies on archived source code and lock hashes.',
            'Only the original six directly archived source files are certified by that source manifest.'])
    assert protected == {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    pd.DataFrame(pairs).to_csv(OUT / 'terminal_sensitivity.csv', index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
