"""Independent source/history, causal price, physical and settlement audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from experiments.exp008.hgb_linked_price_forecast import LinkedPriceForecasts, OUT as PRICE_OUT
from experiments.exp008.closed_loop import optimize
from experiments.exp008.hgb_fixed_mask_mip28_diagnostic import physical_check
from experiments.exp008.planner import execute
from experiments.exp008.verify import verify_npz
from experiments.exp008.run_q4_hgb_linked_price_physical import OUT, ROOT, read_npz, price_causality_check


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(days=334):
    if days not in (3, 334):
        raise ValueError('Only the predetermined first3 or full334 audit is permitted')
    directory = OUT / ('feasibility3' if days == 3 else 'full334')
    archive = directory / 'dispatch_4-2.npz'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    detail = read_npz(archive)
    check = verify_npz(archive, scenario='4-2', expected_days=days,
        initial_soc=protocol['initial_soc'], initial_mode=protocol['initial_mode'],
        initial_power_kw=protocol['initial_power_kw'], audit_path=directory / 'audit.json')
    assert check['passed'], check['errors']
    forecast = LinkedPriceForecasts()
    with np.load(PRICE_OUT/'price_predictions.npz', allow_pickle=False) as price_pack:
        expected_issued_price = price_pack['linked_hgb'].copy()
    maximum_prior_price_difference = 0.
    history_rows, refinement_rows = [], []
    expected_mode = protocol['initial_mode']
    audited_days = (31, 33) if days == 3 else (31, 90, 151, 243, 364)
    for i, day in enumerate(range(31, 31 + days)):
        path = OUT / 'daily_chunks' / f'planning_day_{day}.npz'
        planning = read_npz(path)
        assert int(planning['initial_mode']) == expected_mode
        issue = forecast.get(day, scenario='4-2')
        history = forecast.net_error_paths(day, '4-2', limit=28)
        expected_paths = (issue['load_kw'] - issue['pv_kw'])[None] / 6 + history['errors_kwh']
        np.testing.assert_array_equal(planning['all_net_paths'], expected_paths)
        np.testing.assert_array_equal(planning['predicted_price'], issue['price'])
        np.testing.assert_array_equal(np.column_stack((issue['load_kw'], issue['pv_kw'])),
                                      forecast.store.get(day * 144))
        # Current issued price equals the predeclared forecast diagnostic;
        # the old-HGB comparison changes only price, not net-demand paths.
        prior = read_npz(ROOT / 'data/results/exp008/q4_absolute_hgb_physical/daily_chunks' / f'planning_day_{day}.npz')
        np.testing.assert_array_equal(planning['all_net_paths'], prior['all_net_paths'])
        np.testing.assert_array_equal(planning['predicted_price'], expected_issued_price[day-31])
        np.testing.assert_array_equal(planning['all_price_error_paths'], history['price_errors'])
        independently_rebuilt_price_errors=[]
        for old in history['origins']//144:
            price_prediction, price_audit = forecast._price(int(old)*144,(int(old)+1)*144)
            assert price_audit['price_last_label'] < int(old)*144
            independently_rebuilt_price_errors.append(
                forecast.data.actual[int(old)*144:(int(old)+1)*144,2]-price_prediction)
        np.testing.assert_array_equal(planning['all_price_error_paths'], independently_rebuilt_price_errors)
        maximum_prior_price_difference=max(maximum_prior_price_difference,
            float(np.max(np.abs(prior['predicted_price']-planning['predicted_price']))))
        assert issue['audit']['price_last_label'] < day * 144
        assert not issue['audit']['known_future_price']
        assert history['audit']['max_observed_index'] < day * 144
        selected = np.linspace(0, len(expected_paths) - 1, min(3, len(expected_paths))).astype(int)
        np.testing.assert_array_equal(planning['selected_scenario_indices'], selected)
        np.testing.assert_array_equal(planning['scenario_states'][:, 0],
                                      np.full(len(selected), float(planning['initial_soc'])))
        np.testing.assert_array_equal(planning['allowed_charge'], detail['allowed_charge'][i])
        mask = planning['allowed_charge']
        assert all(np.all(mask[start:start + 6] == mask[start]) for start in range(0, 144, 6))
        scenario_check = physical_check(planning['initial_purchase'], expected_paths[selected],
            planning['allowed_charge'].astype(bool), planning['scenario_states'], planning['scenario_charge'],
            planning['scenario_discharge'], planning['scenario_emergency'], planning['scenario_surplus'])
        assert scenario_check['passed'], scenario_check
        q = planning['refined_purchase']
        np.testing.assert_array_equal(q, detail['original'][i])
        soc = float(planning['initial_soc'])
        assert soc == detail['states'][i, 0]
        observed = forecast.data.actual[day * 144:(day + 1) * 144]
        np.testing.assert_array_equal(detail['actual'][i], observed)
        np.testing.assert_array_equal(detail['price'][i], observed[:, 2])
        reproduced = execute(q, observed[:, :2], planning['predicted_price'], soc, charge_deadband=0.,
                             charge_mask=planning['allowed_charge'])
        for key, value in reproduced.items():
            np.testing.assert_array_equal(value, detail[key][i])
        actual_fees = np.stack((q * observed[:, 2], np.zeros(144), np.zeros(144),
                               5 * reproduced['emergency'] * observed[:, 2]), axis=-1)
        np.testing.assert_array_equal(actual_fees, detail['fees'][i])
        nonzero = np.sign(reproduced['charge'] - reproduced['discharge'])
        nonzero = nonzero[nonzero != 0]
        if len(nonzero):
            expected_mode = int(nonzero[-1])
        history_rows.append({'day': day, 'same_HGB_midnight_and_history_paths_exact': True,
            'own_issued_linked_HGB_price_forecast_exact': True, 'old_HGB_net_paths_unchanged': True, 'own_historical_price_errors_exact': True, 'price_information_boundary_passed': True,
            'scenario_physics_passed': True, 'actual_greedy_arrays_exact': True,
            'actual_ten_minute_transaction_price_settlement_exact': True,
            'scenario_history_source': history['audit']['source'], 'planning_archive_sha256': digest(path)})
        if day in audited_days:
            refined = optimize(planning['initial_purchase'], expected_paths, planning['predicted_price'], soc,
                charge_mask=planning['allowed_charge'], throughput=.002, variation=0.,
                terminal=0. if day == 364 else .45, maxiter=120, deadband=0.)
            np.testing.assert_array_equal(q, refined['purchase'])
            refinement_rows.append({'day': day, 'purchase_refinement_reconstruction_exact': True,
                                    'optimizer_price_is_causal_midnight_forecast': True})
    inputs_unchanged = all(digest(ROOT / path) == expected for path, expected in protocol['source_hashes'].items())
    inputs_unchanged &= all(digest(path) == expected for path, expected in protocol['forecast_artifact_sha256'].items())
    inputs_unchanged &= digest(ROOT / 'data/results/exp008/q4_absolute_hgb_physical/full334/dispatch_4-2.npz') == protocol['baseline_sha256']
    assert inputs_unchanged
    mutation = price_causality_check(forecast)
    result = {'days': days, 'passed': True, 'independent_physical_and_settlement': check,
        'all_days': history_rows, 'refinement_reconstruction': refinement_rows,
        'fresh_future_actual_mutation_test': mutation, 'input_sources_unchanged': bool(inputs_unchanged),
        'model_id': forecast.store.name, 'price_model': 'Ridge28_plus_own_issued_HGB_load_PV',
        'maximum_price_forecast_change_from_baseline': maximum_prior_price_difference, 'this_is_not_a_Q2_8pct_goal_test': True,
        'audit_source_sha256': digest(__file__), 'dispatch_sha256': digest(archive)}
    (directory / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'days': days, 'passed': True, 'cost': check['recomputed_total_cost'],
                      'reversals': check['battery_metrics']['direction_reversals']}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=334)
    run(parser.parse_args().days)
