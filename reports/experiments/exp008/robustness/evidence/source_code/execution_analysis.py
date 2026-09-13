"""Conditional execution stress tests of the accepted Q2 purchase/mode schedule.

This is a frozen-schedule replay, not a rerun of the full forecasting/planning
policy. Every replay carries its own SOC through the entire 334-day period.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/results/exp008/robustness/execution"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(q, mask, actual, initial_soc, capacity=12000., power=5000., efficiency=.9):
    """Same greedy physical control as accepted execution, parameterized hardware."""
    q, mask, actual = q.ravel(), mask.ravel(), actual.reshape(-1, 2)
    eta, low, high, limit = np.sqrt(efficiency), .1*capacity, .9*capacity, power/6
    n = len(q)
    c, d, e, w = (np.zeros(n) for _ in range(4))
    s = np.empty(n+1)
    s[0] = initial_soc
    for t in range(n):
        balance = q[t] + (actual[t, 1]-actual[t, 0])/6
        if balance >= 0:
            c[t] = min(balance, limit, max(0., (high-s[t])/eta)) if mask[t] else 0.
            w[t] = balance-c[t]
        else:
            d[t] = min(-balance, limit, max(0., (s[t]-low)*eta)) if not mask[t] else 0.
            e[t] = -balance-d[t]
        s[t+1] = s[t]+eta*c[t]-d[t]/eta
    balance_error = q+(actual[:, 1]-actual[:, 0])/6+d+e-c-w
    state_error = np.diff(s)-eta*c+d/eta
    checks = dict(balance_error_kwh=float(np.abs(balance_error).max()),
                  soc_error_kwh=float(np.abs(state_error).max()),
                  simultaneous_slots=int(np.sum((c > 1e-6) & (d > 1e-6))),
                  emergency_charge_slots=int(np.sum((c > 1e-6) & (e > 1e-6))),
                  min_soc_kwh=float(s.min()), max_soc_kwh=float(s.max()),
                  max_power_kw=float(max(c.max(), d.max())*6))
    checks['physics_passed'] = bool(max(checks['balance_error_kwh'], checks['soc_error_kwh']) < 1e-6
        and checks['simultaneous_slots'] == checks['emergency_charge_slots'] == 0
        and s.min() >= low-1e-6 and s.max() <= high+1e-6
        and checks['max_power_kw'] <= power+1e-6)
    assert checks['physics_passed'], checks
    return dict(charge=c, discharge=d, emergency=e, surplus=w, states=s), checks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = ROOT/'experiments/exp008/final_selection.json'
    selected = json.loads(selection_path.read_text())['scenarios']['2']
    archive = ROOT/selected['archive']
    with np.load(archive) as z:
        a = {k: v.copy() for k, v in z.items()}
    initial = selected['initial_soc_kwh']
    q, mask, truth, p = a['original'], a['allowed_charge'], a['actual'], a['price']
    assert np.array_equal(q, a['final'])
    raw_paths = [ROOT/'data/raw'/name for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')]
    raw = np.stack([pd.read_csv(f).iloc[31:, 1:].to_numpy(float) for f in raw_paths], axis=-1)
    np.testing.assert_array_equal(raw, truth)
    tariff_path = ROOT/'data/raw/附件1.csv'
    tariff = pd.read_csv(tariff_path)['电价'].to_numpy(float)
    np.testing.assert_array_equal(p, np.broadcast_to(tariff, p.shape))
    raw_paths.append(tariff_path)
    source_paths = [selection_path, archive, *raw_paths, Path(__file__)]
    before = {str(f.relative_to(ROOT)): sha(f) for f in source_paths}
    protocol = dict(scope='Q2 frozen issued purchases and mode masks; own continuous SOC feedback replay',
        dates=['2025-02-01', '2025-12-31'], n_days=334, n_slots=48096,
        reoptimized=False, refitted=False, baseline_schedule_acceptance_unchanged=True,
        biases=[-.10, -.05, -.02, 0., .02, .05, .10],
        hardware=dict(capacity_factors=[.90, .95, 1., 1.05, 1.10],
                      power_factors=[.90, .95, 1., 1.05, 1.10],
                      round_trip_efficiency=[.85, .875, .9, .925, .95]),
        noise=dict(relative_log_std=[.01, .02, .05, .10], replicates=30,
                   seed=20260913, log_noise_AR1=.95, cross_channel_correlation=-.3,
                   construction='Stationary correlated AR(1) Gaussian z; multiply actual by exp(sigma*z - sigma^2/2). PV zeros preserved. Common random numbers across levels.',
                   assumption='Synthetic stress law, not estimated weather uncertainty or confidence interval'),
        capacity_initial_state='Same initial SOC fraction as accepted trajectory; other changes retain initial kWh',
        limitations=['Future daily schedules remain archived even though earlier perturbed history differs. This isolates execution robustness and is not end-to-end policy performance.',
            'Unlimited emergency purchase guarantees load service in this model; physical feasibility alone does not imply economic robustness.',
            'Costs under perturbed inputs are compared to unperturbed exp008 only, not to unperturbed exp006.',
            '2025 data were repeatedly used for model development; this is not an untouched external test.'],
        sources_sha256=before)
    (OUT/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    base, checks = replay(q, mask, truth, initial)
    diffs = {}
    for k in ('charge', 'discharge', 'emergency', 'surplus'):
        diffs[k] = float(np.abs(base[k]-a[k].ravel()).max())
        np.testing.assert_allclose(base[k], a[k].ravel(), atol=1e-9, rtol=0)
    old_s = np.r_[a['states'][0, 0], a['states'][:, 1:].ravel()]
    np.testing.assert_allclose(base['states'], old_s, atol=1e-9, rtol=0)
    planned = float((q*p).sum())
    base_cost = planned+5*float(np.dot(base['emergency'], p.ravel()))
    assert abs(base_cost-a['fees'].sum()) < 1e-6
    rows, daily_rows = [], []

    def evaluate(family, level, actual=truth, replicate_id=-1, **hardware):
        start = initial*hardware.get('capacity', 12000.)/12000.
        result, verified = replay(q, mask, actual, start, **hardware)
        fees = q*p + 5*result['emergency'].reshape(q.shape)*p
        power = 6*(result['charge']-result['discharge'])
        active = np.sign(power[np.abs(power) > 6e-6])
        row = dict(family=family, level=level, replicate_id=replicate_id, n_days=334,
            total_cost_yuan=float(fees.sum()), cost_change_pct=100*(fees.sum()/base_cost-1),
            emergency_cost_yuan=float(fees.sum()-planned), emergency_kwh=float(result['emergency'].sum()),
            peak_emergency_kw=float(6*result['emergency'].max()),
            throughput_kwh=float(result['charge'].sum()+result['discharge'].sum()),
            reversals=int(np.sum(active[1:] != active[:-1])), final_soc_kwh=float(result['states'][-1]),
            **verified)
        rows.append(row)
        for i, cost in enumerate(fees.sum(axis=1)):
            daily_rows.append(dict(family=family, level=level, replicate_id=replicate_id, day=i+31, total_cost_yuan=float(cost)))
        if family == 'baseline' or (family == 'combined_bias' and level in (-.1, .1)):
            np.savez_compressed(OUT/f'{family}_{level:+.2f}.npz', **result, actual=actual, fees=fees)

    evaluate('baseline', 0.)
    for family in ('load_bias', 'pv_bias', 'combined_bias'):
        for bias in protocol['biases']:
            changed = truth.copy()
            if family in ('load_bias', 'combined_bias'):
                changed[:, :, 0] *= 1+bias
            if family == 'pv_bias':
                changed[:, :, 1] *= 1+bias
            elif family == 'combined_bias':
                changed[:, :, 1] *= 1-bias
            evaluate(family, bias, changed)
    for ratio in protocol['hardware']['capacity_factors']:
        evaluate('capacity', ratio, capacity=12000.*ratio)
    for ratio in protocol['hardware']['power_factors']:
        evaluate('power', ratio, power=5000.*ratio)
    for efficiency in protocol['hardware']['round_trip_efficiency']:
        evaluate('efficiency', efficiency, efficiency=efficiency)
    rng = np.random.default_rng(protocol['noise']['seed'])
    rho = protocol['noise']['cross_channel_correlation']
    phi = protocol['noise']['log_noise_AR1']
    for seed in range(30):
        eps = rng.normal(size=(len(q.ravel()), 2))
        eps[:, 1] = rho*eps[:, 0]+np.sqrt(1-rho*rho)*eps[:, 1]
        eps[0] /= np.sqrt(1-phi*phi)
        z = lfilter([np.sqrt(1-phi*phi)], [1., -phi], eps, axis=0).reshape(truth.shape)
        for sigma in protocol['noise']['relative_log_std']:
            evaluate('correlated_noise', sigma, truth*np.exp(sigma*z-.5*sigma*sigma), replicate_id=seed)
        if (seed+1) % 10 == 0:
            print(f'NOISE_REPLICATES {seed+1}/30', flush=True)
    # The time-local executor cannot depend on future observations.
    changed = truth.copy().reshape(-1, 2)
    cutoff = 100*144+37
    changed[cutoff:] *= [1.2, .8]
    mutated, _ = replay(q, mask, changed, initial)
    for key in ('charge', 'discharge', 'emergency', 'surplus'):
        np.testing.assert_array_equal(mutated[key][:cutoff], base[key][:cutoff])
    np.testing.assert_array_equal(mutated['states'][:cutoff+1], base['states'][:cutoff+1])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'stress_metrics.csv', index=False)
    pd.DataFrame(daily_rows).to_csv(OUT/'stress_daily.csv', index=False)
    noise_rows=[]
    for sigma, group in frame[frame.family == 'correlated_noise'].groupby('level'):
        values=group.cost_change_pct
        noise_rows.append(dict(log_noise_std=sigma, replicates=len(group),
            mean_cost_change_pct=float(values.mean()), min_cost_change_pct=float(values.min()),
            p05_cost_change_pct=float(values.quantile(.05)), p50_cost_change_pct=float(values.median()),
            p95_cost_change_pct=float(values.quantile(.95)), max_cost_change_pct=float(values.max())))
    pd.DataFrame(noise_rows).to_csv(OUT/'noise_summary.csv', index=False)
    assert all(sha(ROOT/f) == h for f,h in before.items())
    summary=dict(passed=True, n_scenarios=len(frame), baseline_cost_yuan=base_cost,
        baseline_max_abs_array_differences=diffs, all_physics_passed=bool(frame.physics_passed.all()),
        future_actual_prefix_mutation_passed=True, source_hashes_unchanged=True,
        noise=noise_rows, largest_cost_change=frame.loc[frame.cost_change_pct.idxmax()].to_dict(),
        interpretation=protocol['limitations'])
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
