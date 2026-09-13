"""Two fixed-Q perfect-information bounds; no future action trace is exported."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from experiments.exp008.fixed_purchase_bounds import sha256, solve_fixed_purchase

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/results/exp008/cumulative_risk_planning'
OUT = ROOT/'data/results/exp008/oracle_diagnostic_cumulative_risk'
TARGET = 12940956.8791
NAMES = ('prefix', 'point')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def inventory():
    return {str(path.relative_to(SOURCE)): sha256(path) for path in sorted(SOURCE.rglob('*')) if path.is_file()}


def run():
    if OUT.exists():
        raise FileExistsError('Do not overwrite another fixed-purchase bound evaluation')
    OUT.mkdir(parents=True)
    protected = inventory()
    source_code = [Path(__file__), ROOT/'experiments/exp008/fixed_purchase_bounds.py',
                   ROOT/'experiments/exp008/verify.py']
    code_hashes = {}
    for path in source_code:
        destination = OUT/'source_archive'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        code_hashes[str(path.relative_to(ROOT))] = sha256(destination)
    protocol = {'fixed_names': NAMES, 'maximum_configurations': 2, 'days': 334,
        'selection_predefined_by_parent': True, 'source_tree_sha256_before': protected,
        'source_code_sha256': code_hashes, 'fixed_Q_from_original_not_reoptimized': True,
        'solver': 'existing fixed_purchase_bounds.solve_fixed_purchase annual physical LP',
        'information': 'all future actual net demand; noncausal diagnostic only',
        'physical': 'continuous annual SOC, original efficiency/capacity/power, no emergency charging or simultaneous charge/discharge',
        'relaxations': 'real-time state_buffer, deadband, mode/ramp restrictions and switching budget are not imposed',
        'not_an_isolated_state_buffer_ablation': True,
        'oracle_actions_exported_or_reused': False, 'eligible_for_dispatch_index': False,
        'cost_target_yuan': TARGET}
    save(OUT/'protocol.json', protocol)
    results = []
    for name in NAMES:
        report = solve_fixed_purchase(SOURCE/name/'dispatch_2.npz', OUT/name, quantile=None)
        assert report['days'] == 334 and report['source_verified']
        assert report['primal_dual_gap_yuan'] < 1e-4 and report['dual_stationarity_max_error'] < 1e-6
        assert report['max_constraint_residual_kwh'] < 1e-6
        results.append({'method': name, 'actual_cost_yuan': report['candidate_actual_total_cost_yuan'],
            'fixed_planned_cost_yuan': report['frozen_planned_cost_yuan'],
            'actual_emergency_cost_yuan': report['candidate_actual_emergency_cost_yuan'],
            'oracle_minimum_emergency_cost_yuan': report['minimum_emergency_cost_yuan'],
            'oracle_minimum_total_cost_yuan': report['minimum_total_cost_yuan'],
            'total_dual_bound_yuan': report['total_dual_bound_yuan'],
            'maximum_execution_only_saving_yuan': report['maximum_saving_by_execution_only_yuan'],
            'oracle_minus_8pct_target_yuan': report['minimum_total_cost_yuan']-TARGET,
            'frozen_Q_execution_only_8pct_ruled_out_by_dual': report['total_dual_bound_yuan'] > TARGET,
            'primal_dual_gap_yuan': report['primal_dual_gap_yuan'],
            'max_constraint_residual_kwh': report['max_constraint_residual_kwh'],
            'source_sha256': report['source_sha256']})
    assert inventory() == protected, 'Protected source or original results changed'
    assert not list(OUT.rglob('*.npz')), 'Never export an oracle action trace'
    pd.DataFrame(results).to_csv(OUT/'comparison.csv', index=False)
    prefix, point = results
    summary = {'complete': True, 'bounds': results, 'source_tree_hashes_unchanged': True,
        'oracle_actions_exported': False, 'added_to_dispatch_index': False,
        'actual_prefix_minus_point_yuan': prefix['actual_cost_yuan']-point['actual_cost_yuan'],
        'oracle_prefix_minus_point_yuan': prefix['oracle_minimum_total_cost_yuan']-point['oracle_minimum_total_cost_yuan'],
        'prefix_extra_execution_headroom_vs_point_yuan': prefix['maximum_execution_only_saving_yuan']-point['maximum_execution_only_saving_yuan'],
        'interpretation': 'A dual bound above the target rules out reaching it with the frozen Q even with perfect future execution. A lower bound below the target establishes only that impossibility is not certified; it does not construct a causal or low-switch controller.',
        'state_buffer_attribution_limit': 'Execution headroom includes perfect foresight and all timing changes, not only releasing state_buffer500. This is not a causal attribution to that buffer.',
        'no_causal_goal_claim': True}
    save(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
