"""Read-only count and midnight-value diagnostics; no new strategy or plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008/terminal_count_diagnostic'
BASE = ROOT / 'data/results/exp006/primary/dispatch_2.npz'
CAND = ROOT / ('data/results/exp008/mode_budget_hold1_physical/'
               'direct_hgb_memory_hold1_cap8_kappa0_334days/dispatch.npz')
ETA, LOW, HIGH, LIMIT, TOL = np.sqrt(.9), 1200., 10800., 5000/6, 1e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def count(detail):
    """Independent chronological loop, allowing idle to break an episode."""
    c, d = detail['charge'].ravel(), detail['discharge'].ravel()
    previous, last_nonzero = 0, 0
    starts = {1: 0, -1: 0}
    reversals = 0
    for charge, discharge in zip(c, d):
        mode = int(np.sign(charge-discharge)) if abs(charge-discharge) > TOL else 0
        if mode and mode != previous:
            starts[mode] += 1
        if mode:
            if last_nonzero and mode != last_nonzero:
                reversals += 1
            last_nonzero = mode
        previous = mode
    return {'direction_reversals_excluding_warmup': reversals,
            'charge_episodes_excluding_warmup': starts[1],
            'discharge_episodes_excluding_warmup': starts[-1],
            'all_charge_discharge_episodes': sum(starts.values()),
            'active_ten_minute_slots': int(np.sum((c > TOL) | (d > TOL))),
            'throughput_ac_kwh': float(np.sum(c+d)),
            'equivalent_full_cycles': float(np.sum(ETA*c+d/ETA)/24000),
            'simultaneous_slots': int(np.sum((c > TOL) & (d > TOL)))}


def fixed_replay(q, actual, price, mask, initial):
    """Same observation-only greedy executor, independently scalar-coded."""
    s = float(initial)
    c, d, e, states = [], [], [], [s]
    for purchase, values, tariff, can_charge in zip(q, actual, price, mask):
        balance = purchase-(values[0]-values[1])/6
        ct = min(balance, LIMIT, max(0., (HIGH-s)/ETA)) if balance >= 0 and can_charge else 0.
        dt = min(-balance, LIMIT, max(0., (s-LOW)*ETA)) if balance < 0 and not can_charge else 0.
        et = max(0., -balance-dt)
        s += ETA*ct-dt/ETA
        c.append(ct); d.append(dt); e.append(et); states.append(s)
    return np.asarray(c), np.asarray(d), np.asarray(e), np.asarray(states)


def stats(values):
    values = np.asarray(values)
    return {'minimum': float(values.min()), 'mean': float(values.mean()),
            'q25': float(np.quantile(values, .25)), 'median': float(np.median(values)),
            'q75': float(np.quantile(values, .75)), 'maximum': float(values.max())}


def main():
    if OUT.exists():
        raise FileExistsError('Version outputs rather than replacing diagnostic evidence')
    OUT.mkdir(parents=True)
    source = [BASE, CAND, ROOT/'reports/experiments/exp006/report.md',
              ROOT/'experiments/problem2/tree_planning/verify.py',
              ROOT/'experiments/exp008/closed_loop.py',
              ROOT/'experiments/exp008/run_hgb_mode_budget_hold1.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source}
    protocol = {'diagnostic_only': True, 'sources_sha256': hashes,
        'no_new_purchase_plans_or_annual_policy_run': True,
        'fixed_initial_inventory_probes_kwh': [100., 1000.],
        'probe_uses_realized_future_for_retrospective_evaluation': True,
        'probe_policy_selected_using_future': False,
        'probe_start_SOC_is_free_intervention_not_reachable_continuous_dispatch': True,
        'probe_actions_not_exported': True,
        'source_sha256': digest(__file__)}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    a, b = read(BASE), read(CAND)
    ca, cb = count(a), count(b)
    metrics = pd.DataFrame([{'metric': key, 'exp006': ca[key], 'hold1_cap8': cb[key],
        'delta': cb[key]-ca[key], 'change_pct': 100*(cb[key]/ca[key]-1) if ca[key] else 0.}
        for key in ca])
    metrics.to_csv(OUT/'count_comparison.csv', index=False)
    assert ca['direction_reversals_excluding_warmup'] == 2729
    assert ca['all_charge_discharge_episodes'] == 3791
    assert cb['direction_reversals_excluding_warmup'] == 2521
    assert cb['all_charge_discharge_episodes'] == 3888
    s, p, e = b['states'], b['price'], b['emergency']
    # Exclude Feb 1, whose predecessor is the shared January warmup boundary.
    next_days = np.arange(1, 334)
    rows, probe_rows = [], []
    maximum_rebuild_error = 0.
    for i in next_days:
        q, actual, mask = b['original'][i], b['actual'][i], b['allowed_charge'][i]
        rc, rd, re, rs = fixed_replay(q, actual, p[i], mask, s[i, 0])
        for computed, stored in ((rc,b['charge'][i]),(rd,b['discharge'][i]),
                                 (re,e[i]),(rs,s[i])):
            maximum_rebuild_error = max(maximum_rebuild_error, float(np.max(np.abs(computed-stored))))
        row = {'day': int(i+31), 'date': str((pd.Timestamp('2025-02-01')+pd.Timedelta(days=int(i))).date()),
               'initial_soc_kwh': float(s[i,0]), 'previous_day_end_soc_kwh': float(s[i-1,-1]),
               'next_day_end_soc_kwh': float(s[i,-1]),
               'emergency_fee_00_06_yuan': float(np.dot(5*p[i,:36],e[i,:36])),
               'emergency_fee_00_09_yuan': float(np.dot(5*p[i,:54],e[i,:54])),
               'emergency_fee_21_24_yuan': float(np.dot(5*p[i,126:],e[i,126:])),
               'emergency_fee_full_day_yuan': float(np.dot(5*p[i],e[i])),
               'original_purchase_fee_00_06_yuan': float(np.dot(p[i,:36],q[:36]))}
        rows.append(row)
        for extra in (100., 1000.):
            added = min(extra, HIGH-s[i,0])
            pc, pd_, pe, ps = fixed_replay(q, actual, p[i], mask, s[i,0]+added)
            assert np.min(re-pe) >= -1e-6
            probe_rows.append({'day': int(i+31), 'extra_inventory_requested_kwh': extra,
                'extra_inventory_actual_kwh': float(added),
                'conditional_emergency_saving_00_06_yuan': float(np.dot(5*p[i,:36],re[:36]-pe[:36])),
                'conditional_emergency_saving_full_day_yuan': float(np.dot(5*p[i],re-pe)),
                'residual_inventory_at06_kwh': float(ps[36]-rs[36]),
                'residual_inventory_dayend_kwh': float(ps[-1]-rs[-1]),
                'notional_added_inventory_value_at_045_yuan': .45*float(added)})
    assert maximum_rebuild_error < 1e-8
    frame, probes = pd.DataFrame(rows), pd.DataFrame(probe_rows)
    frame.to_csv(OUT/'boundary_daily_diagnostic.csv',index=False)
    probes.to_csv(OUT/'conditional_fixed_purchase_inventory_probes.csv',index=False)
    probe_summary = {}
    for extra, group in probes.groupby('extra_inventory_requested_kwh'):
        vals = group['conditional_emergency_saving_full_day_yuan']/group['extra_inventory_actual_kwh']
        six = group['conditional_emergency_saving_00_06_yuan']/group['extra_inventory_actual_kwh']
        probe_summary[str(extra)] = {
            'mean_conditional_full_day_fee_value_yuan_per_soc_kwh': float(vals.mean()),
            'mean_conditional_00_06_fee_value_yuan_per_soc_kwh': float(six.mean()),
            'full_day_value_distribution': stats(vals),
            'days_full_day_value_exceeds045': int((vals > .45+1e-8).sum()),
            'days_extra_soc_has_zero_full_day_emergency_value': int((vals < 1e-8).sum()),
            'mean_inventory_fraction_remaining_at06': float(np.mean(group['residual_inventory_at06_kwh']/group['extra_inventory_actual_kwh'])),
            'mean_inventory_fraction_remaining_at_dayend': float(np.mean(group['residual_inventory_dayend_kwh']/group['extra_inventory_actual_kwh']))}
    annual_emergency = float(np.sum(5*p*e))
    morning = float(frame['emergency_fee_00_06_yuan'].sum())
    late = float(frame['emergency_fee_21_24_yuan'].sum())
    total = float(b['fees'].sum())
    result = {'passed': True, 'original_sources_unchanged': all(digest(ROOT/name)==h for name,h in hashes.items()),
        'independent_executor_maximum_rebuild_error_kwh': maximum_rebuild_error,
        'historical_count_source': {'report': 'reports/experiments/exp006/report.md:105',
            'code': 'experiments/problem2/tree_planning/verify.py:21',
            'primary_historical_metric': 'direction changes in chronological non-idle signs; idle does not reset last sign',
            'episode_metric': 'each charge/discharge stretch separated by idle or other direction; evaluation boundary treated idle'},
        'counts': {'exp006': ca, 'hold1_cap8': cb},
        'unqualified_total_operation_count_reduction_supported': False,
        'terminal_SOC_scope': {'boundaries': 333, 'exclude_final_evaluation_day_terminal_state': True,
            'actual_boundary_soc_kwh': stats(s[:-1,-1]), 'boundary_soc_at_physical_lower_bound': int((s[:-1,-1]<=LOW+TOL).sum()),
            'boundary_soc_within50kwh_lower_bound': int((s[:-1,-1]<=LOW+50).sum()),
            'night_emergency_00_06_yuan': morning, 'night_emergency_00_06_share_of_full334_emergency_pct': 100*morning/annual_emergency,
            'night_emergency_00_06_for_soc_within50kwh_low_yuan': float(frame.loc[frame['initial_soc_kwh']<=LOW+50,'emergency_fee_00_06_yuan'].sum()),
            'night_emergency_00_09_yuan': float(frame['emergency_fee_00_09_yuan'].sum()),
            'late_emergency_21_24_yuan': late,
            'annual_emergency_yuan': annual_emergency,
            'gap_to_8pct_yuan': total-.92*float(a['fees'].sum()),
            '00_06_entire_emergency_share_of_gap_pct': 100*morning/(total-.92*float(a['fees'].sum()))},
        'terminal_unit_economics': {'units': 'yuan per battery SOC kWh; not AC kWh',
            'greedy_terminal_value': .45, 'MIP_terminal_value_min_tariff_over_eta': float(p.min()/ETA),
            'fixed_tariff_minmax_yuan_per_ac_kwh': [float(p.min()),float(p.max())],
            'overnight_00_06_replenishment_cost_per_soc_kwh': stats(p[0,:36]/ETA),
            'emergency_displacement_value_00_06_per_soc_kwh': stats(5*p[0,:36]*ETA),
            'both_terminal_values_zero_on_final_day': True,
            'reported_actual_fee_contains_no_terminal_credit': True},
        'conditional_probes': probe_summary,
        'probe_scope': 'Free independent extra initial SOC, fixed purchases and mode masks. Uses realized demand only for retrospective evaluation. Not an implementable continuous annual strategy, not additive feasible savings, not optimized next-day purchase value.',
        'two_day_horizon_assessment': 'Night deficit alone is too small to explain the 8% gap. Two-day planning could change both next-day purchases and overflow, which these fixed-Q probes cannot price. A small causally constructed two-day-horizon diagnostic may test model value, but evidence does not justify an immediate full annual run or claim that terminal correction will close the gap.'}
    assert result['original_sources_unchanged']
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
