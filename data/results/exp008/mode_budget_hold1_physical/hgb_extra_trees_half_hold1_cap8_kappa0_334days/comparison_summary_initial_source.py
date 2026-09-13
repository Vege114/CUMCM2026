"""Compact, source-backed comparison after the predeclared 334-day run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.run_blend_mode_budget_hold1 import OUT, HGB_HOLD1, digest, save


def summarize():
    completed = json.loads((OUT/'summary.json').read_text())
    audited = json.loads((OUT/'independent_audit.json').read_text())
    assert completed['completed_days'] == audited['days'] == 334 and audited['passed']
    assert json.loads((OUT/'runtime_source_consistency_after.json').read_text())['passed']
    findings = json.loads((OUT/'paired_findings.json').read_text())
    metrics = {m['name']:m for m in findings['metrics']}
    candidate = metrics['hgb_extra_trees_half_hold1_cap8_kappa0']
    baseline = metrics['exp006']
    previous = metrics['original_HGB_hold1_cap8_kappa0']
    old_sources = json.loads((HGB_HOLD1/'complete_provenance.json').read_text())['source_sha256']
    new_sources = json.loads((OUT/'complete_provenance.json').read_text())['source_sha256']
    numerical_paths = ['experiments/exp008/mode_budget_hold1_physical.py',
                       'experiments/exp008/closed_loop.py',
                       'experiments/exp008/planner.py',
                       'experiments/common/neural_v2/physics.py']
    algorithm_hashes = {p:new_sources[p] for p in numerical_paths}
    assert all(new_sources[p] == old_sources[p] for p in numerical_paths)
    audits = json.loads((OUT/'planning_audit.json').read_text())
    gaps = np.array([a['mip']['mip_gap'] for a in audits])
    comparison_keys = ['total_cost','planned_cost','emergency_cost','direction_reversals',
        'charge_starts','discharge_starts','actual_contiguous_active_segments',
        'actual_contiguous_active_segments_1slots','actual_contiguous_active_segments_2slots',
        'actual_contiguous_active_segments_3slots','planned_mode_segments_1slots',
        'planned_mode_segments_2slots','planned_mode_segments_3slots','active_slots',
        'throughput_kwh','equivalent_full_cycles','power_ramp_total_kw','final_soc']
    def differences(other):
        return {key:candidate[key]-other[key] for key in comparison_keys
                if candidate[key] is not None and other[key] is not None}
    result = {'completed_days':334,'candidate':candidate,
        'same_fixed_planning_algorithm_source_sha256':algorithm_hashes,
        'only_declared_model_input_change_is_forecast_and_own_issued_history':True,
        'timed_MIP_limitation':'Same5secondlimit and targetgap; independent timed solver runs can return different feasible incumbents. This is not a guarantee of deterministic optimizer replay.',
        'delta_vs_original_HGB_hold1':differences(previous),
        'delta_vs_exp006':differences(baseline),
        'cost_reduction_vs_exp006_pct':100*(1-candidate['total_cost']/baseline['total_cost']),
        'cost_gap_to8pct_yuan':candidate['total_cost']-.92*baseline['total_cost'],
        'battery_intensity_metrics_are_not_a_lifetime_estimate':True,
        'mip_gaps':{'mean':float(gaps.mean()),'max':float(gaps.max()),
                    'over005':int(np.sum(gaps>.005)), 'over01':int(np.sum(gaps>.01)),
                    'feasible_days':334},
        'greedy_refinement_success_days':sum(bool(a['greedy_refinement']['success']) for a in audits),
        'greedy_refinement_max_iterations_per_day':120,
        'first3_prefix_equal':all(audited['prefix_arrays_equal'].values()),
        'goal':findings['goal'],'development_not_independent_test':True,
        'final_model_selection':False,'source_sha256':digest(Path(__file__))}
    monthly = pd.read_csv(OUT/'monthly_comparison.csv')
    selected = monthly[monthly['name'] == 'hgb_extra_trees_half_hold1_cap8_kappa0'].set_index('month')
    old = monthly[monthly['name'] == 'original_HGB_hold1_cap8_kappa0'].set_index('month')
    columns = ['total_cost','emergency_cost','direction_reversals','active_slots',
               'throughput_kwh','power_ramp_total_kw','final_soc']
    (selected[columns]-old[columns]).to_csv(OUT/'monthly_deltas_vs_HGB_hold1.csv')
    save(OUT/'matched_HGB_hold1_effect_summary.json',result)
    (OUT/'comparison_summary_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    summarize()
