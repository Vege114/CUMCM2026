"""Independent envelope identity, empirical coverage and all-slot floor audit."""
import json
import shutil
import numpy as np

from experiments.exp008.simultaneous_prefix_envelope import OUT, ROOT, digest, save
from experiments.exp008.controller_candidate import ETA, MIN_SOC, POWER_ENERGY


def main():
    with np.load(OUT / 'risk_inputs.npz') as z:
        inputs = {key: z[key].copy() for key in z.files}
    with np.load(OUT / 'dispatch_2.npz') as z:
        actual = {key: z[key].copy() for key in z.files}
    with np.load(OUT / 'floor_diagnostics.npz') as z:
        floors = {key: z[key].copy() for key in z.files}
    rows = []
    for i, day in enumerate(inputs['days']):
        cumulative = np.cumsum(inputs['errors'][i], axis=1)
        mean = np.sum(cumulative, axis=0) / 28
        sigma = np.maximum(np.sqrt(np.sum((cumulative - mean) ** 2, axis=0) / 28), 10.)
        score = np.max((cumulative - mean) / sigma, axis=1)
        ordered = np.sort(score)
        index = .8 * 27
        lo, frac = int(np.floor(index)), index - np.floor(index)
        threshold = ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])
        expected = mean + threshold * sigma
        np.testing.assert_allclose(inputs['envelope'][i], expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(inputs['score_quantile'][i], threshold, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.cumsum(inputs['risks'][i] - inputs['forecast_net'][i]), expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.sum(inputs['risks'][i]) - np.sum(inputs['forecast_net'][i]), expected[-1], rtol=0, atol=1e-8)
        all_prefixes = np.all(cumulative <= expected + 1e-8, axis=1)
        np.testing.assert_array_equal(all_prefixes, inputs['history_simultaneously_covered'][i])
        assert all_prefixes.sum() == 22
        intended = actual['intended_states'][i]
        risk_balance = actual['original'][i] - actual['intended_charge'][i] + actual['intended_discharge'][i] - inputs['risks'][i]
        assert risk_balance.min() >= -1e-7  # remaining nonnegative planned spill
        intended_soc_error = np.diff(intended) - ETA * actual['intended_charge'][i] + actual['intended_discharge'][i] / ETA
        assert np.abs(intended_soc_error).max() < 1e-7
        # Scalar independently derived floor and immediate withheld demand.
        extra = np.zeros(144)
        for slot in range(144):
            q = actual['original'][i, slot]
            observed = actual['actual'][i, slot]
            shortage = max(0., (observed[0] - observed[1]) / 6 - q)
            before = actual['states'][i, slot]
            floor = max(MIN_SOC, intended[slot + 1] - 500.)
            without_floor = min(shortage, POWER_ENERGY, max(0., (before - MIN_SOC) * ETA))
            with_floor = min(shortage, POWER_ENERGY, max(0., (before - floor) * ETA))
            assert abs(with_floor - actual['discharge'][i, slot]) < 1e-8
            assert floor == floors['floor_kwh'][i, slot]
            extra[slot] = max(0., without_floor - with_floor)
        np.testing.assert_array_equal(extra, floors['extra_unserved_demand_due_floor_kwh'][i])
        rows.append({'day': int(day), 'independent_envelope_and_endpoint_identities_passed': True,
            'empirical_simultaneous_coverage_count': int(all_prefixes.sum()),
            'LP_reference_balance_and_intended_SOC_passed': True,
            'all144_floor_and_immediate_unserved_demand_values_rebuilt': True})
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert all(digest(ROOT / name) == value for name, value in protocol['source_sha256'].items())
    assert all(digest(name) == value for name, value in protocol['input_sha256'].items())
    result = {'passed': True, 'days': 334, 'day_checks': rows,
        'finite_sample_cover_rate_is_22_over28_not_exactly_point8': True,
        'all_negative_risk_increments_preserved': int(np.sum(inputs['increments'] < 0)),
        'all_negative_risk_levels_preserved': int(np.sum(inputs['risks'] < 0)),
        'floor_proxy_is_not_realizable_annual_saving': True,
        'source_and_input_hashes_unchanged': True, 'audit_source_sha256': digest(__file__)}
    save(OUT / 'independent_envelope_floor_audit.json', result)
    shutil.copy2(__file__, OUT / 'independent_audit_source.py')
    print('Passed: 334 envelope/LP/coverage cases and all48096 floor/demand slots independently rebuilt.')


if __name__ == '__main__':
    main()
