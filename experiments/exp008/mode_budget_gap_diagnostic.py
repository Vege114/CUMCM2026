"""Read-only post-run gap decomposition; no optimization or day selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import ETA, HIGH, LIMIT, LOW, objective
from experiments.exp008.run_hgb_mode_budget import HOURLY, OUT


def load(path):
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_terms(q, paths, price, soc, mask, terminal):
    """Independent vector replay of the frozen strict-mask, zero-deadband law."""
    states = np.full(len(paths), soc)
    emergency = np.zeros(len(paths))
    throughput = np.zeros(len(paths))
    for t in range(len(q)):
        balance = q[t] - paths[:, t]
        c = np.minimum.reduce((np.maximum(balance, 0),
            np.full(len(paths), LIMIT), np.maximum(0, (HIGH-states)/ETA))) if mask[t] else np.zeros(len(paths))
        d = np.minimum.reduce((np.maximum(-balance, 0),
            np.full(len(paths), LIMIT), np.maximum(0, (states-LOW)*ETA))) if not mask[t] else np.zeros(len(paths))
        emergency += 5*price[t]*np.maximum(0, -balance-d)
        throughput += c+d
        states += ETA*c-d/ETA
    terms = {'planned_cost': float(q@price), 'expected_emergency_cost': float(emergency.mean()),
        'expected_wear_proxy': float(.002*throughput.mean()),
        'terminal_credit': float(terminal*(states-LOW).mean())}
    value = terms['planned_cost']+terms['expected_emergency_cost']+terms['expected_wear_proxy']-terms['terminal_credit']
    analytic = objective(q, paths, price, soc, charge_mask=mask, throughput=.002,
                         variation=0., terminal=terminal, deadband=0.)[0]
    np.testing.assert_allclose(value, analytic, atol=1e-6, rtol=0.)
    return {'objective': value, **terms}


def run():
    arrays, hourly = load(OUT/'dispatch.npz'), load(HOURLY/'dispatch.npz')
    assert len(arrays['days']) == len(hourly['days']) == 334
    np.testing.assert_array_equal(arrays['days'], hourly['days'])
    audits = json.loads((OUT/'planning_audit.json').read_text())
    rows = []
    for i, day in enumerate(arrays['days']):
        day = int(day)
        p, h = load(OUT/f'planning_day{day}.npz'), load(HOURLY/f'planning_day{day}.npz')
        paths, mask, price = p['all_net_paths'], p['allowed_charge'], p['price']
        np.testing.assert_array_equal(paths, h['all_net_paths'])
        terminal = 0. if day == 364 else .45
        row = {'day': day, 'date': str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
            'mip_gap': audits[i]['mip']['mip_gap'],
            'refinement_success': audits[i]['greedy_refinement']['success'],
            'refinement_iterations': audits[i]['greedy_refinement']['iterations'],
            'candidate_actual_planned_cost': float(arrays['fees'][i, :, 0].sum()),
            'candidate_actual_emergency_cost': float(arrays['fees'][i, :, 3].sum()),
            'candidate_actual_cost': float(arrays['fees'][i].sum()),
            'hourly_actual_cost': float(hourly['fees'][i].sum()),
            'candidate_minus_hourly_actual_cost': float(arrays['fees'][i].sum()-hourly['fees'][i].sum())}
        for label, q in [('initializer', p['purchase']), ('refined', arrays['original'][i])]:
            for count, selected in [(28, paths), (3, paths[p['selected_scenario_indices']])]:
                terms = replay_terms(q, selected, price, float(p['initial_soc']), mask, terminal)
                row.update({f'{label}_historical{count}_{key}': value for key, value in terms.items()})
        archived_objective = audits[i]['greedy_refinement']['objective']
        np.testing.assert_allclose(row['refined_historical28_objective'], archived_objective, atol=1e-6, rtol=0.)
        terms = replay_terms(hourly['original'][i], paths, price, float(h['initial_soc']),
                             hourly['allowed_charge'][i], terminal)
        row.update({f'hourly_own_soc_and_mask_historical28_{key}': value for key, value in terms.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'gap_daily_exact_history_objectives.csv', index=False)
    signed_difference = frame.candidate_minus_hourly_actual_cost
    gross_savings = -signed_difference.clip(upper=0)
    gross_increases = signed_difference.clip(lower=0)
    groups = []
    for threshold in (.005, .01, .05):
        selected = frame.mip_gap > threshold
        group = {'gap_strictly_above': threshold, 'days': int(selected.sum()),
            'candidate_plan_plus_emergency_cost': float(frame.loc[selected, 'candidate_actual_cost'].sum()),
            'share_of_annual_candidate_bill': float(frame.loc[selected, 'candidate_actual_cost'].sum()/frame.candidate_actual_cost.sum()),
            'candidate_planned_cost': float(frame.loc[selected, 'candidate_actual_planned_cost'].sum()),
            'candidate_emergency_cost': float(frame.loc[selected, 'candidate_actual_emergency_cost'].sum()),
            'candidate_minus_hourly_cost': float(signed_difference[selected].sum()),
            'share_of_signed_annual_candidate_minus_hourly_difference': float(signed_difference[selected].sum()/signed_difference.sum()) if signed_difference.sum() else None,
            'gross_savings_share': float(gross_savings[selected].sum()/gross_savings.sum()) if gross_savings.sum() else None,
            'gross_increase_share': float(gross_increases[selected].sum()/gross_increases.sum()) if gross_increases.sum() else None,
            'days_of_year': frame.loc[selected, 'day'].astype(int).tolist()}
        groups.append(group)
    result = {'completed_days': 334, 'post_run_descriptive_only': True, 'no_optimization_or_parameter_changes': True,
        'gap_and_historical_objectives_known_from_previous_complete_days_only': True,
        'actual_fee_grouping_is_retrospective_not_a_day_selection_rule': True,
        'all_history_objectives_replayed_independently_and_match_analytic_objective': True,
        'refined_history28_objective_matches_archived_optimizer': True,
        'history_objective_includes_wear_and_terminal_but_actual_bill_does_not': True,
        'same_HGB_hourly_history_values_equal': True,
        'hourly_objective_uses_its_own_continuous_SOC_and_mask_not_a_pure_Q_ablation': True,
        'signed_share_can_be_negative_or_exceed_one_due_to_offsetting_daily_differences': True,
        'annual_actual_cost': float(frame.candidate_actual_cost.sum()),
        'annual_candidate_minus_hourly_cost': float(signed_difference.sum()),
        'gap_quantiles': {str(q): float(frame.mip_gap.quantile(q)) for q in (0., .5, .9, .95, 1.)},
        'refinement_success_days': int(frame.refinement_success.sum()),
        'summed_initializer_history28_objective': float(frame.initializer_historical28_objective.sum()),
        'summed_refined_history28_objective': float(frame.refined_historical28_objective.sum()),
        'groups': groups,
        'source_sha256': {str(path): sha(path) for path in (Path(__file__),
            Path('experiments/exp008/closed_loop.py'), OUT/'dispatch.npz', OUT/'planning_audit.json',
            HOURLY/'dispatch.npz', OUT/'gap_daily_exact_history_objectives.csv')}}
    (OUT/'gap_cost_decomposition.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    run()
