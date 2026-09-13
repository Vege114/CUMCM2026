"""One explicitly requested fixed-Q oracle bound; no future actions exported."""
import hashlib
import json
from pathlib import Path
import shutil

from experiments.exp008.fixed_purchase_bounds import solve_fixed_purchase
from experiments.exp008.verify import BASELINE_COST, ETA, INITIAL_SOC, MAX_SOC, MIN_SOC, POWER_ENERGY

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/results/exp008/load_energy_memory/lp_bridge/memory_tree28_q08_buffer500_334days/dispatch_2.npz'
OUT = ROOT / 'data/results/exp008/oracle_diagnostic_memory_lp_q08'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError('Oracle diagnostics are immutable; the named diagnostic already exists')
    OUT.mkdir(parents=True)
    protected = ROOT / 'data/results/exp008/fixed_purchase_oracle.json'
    protection = digest(protected)
    sources = [Path(__file__), ROOT / 'experiments/exp008/fixed_purchase_bounds.py',
               ROOT / 'experiments/exp008/verify.py']
    hashes = {}
    for source in sources:
        shutil.copy2(source, OUT / source.name)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    protocol = {'source_fixed_purchase': str(SOURCE), 'source_sha256': digest(SOURCE),
        'selection_rule': 'single explicit parent-requested memory_tree28_q08_buffer500_334days archive; no selection on oracle outcome',
        'days': 334, 'slots': 48096, 'quantile_of_fixed_source': .8,
        'initial_soc_kwh': INITIAL_SOC, 'soc_min_kwh': MIN_SOC, 'soc_max_kwh': MAX_SOC,
        'capacity_kwh': 12000., 'eta_charge': ETA, 'eta_discharge': ETA,
        'max_charge_discharge_power_kw': 6 * POWER_ENERGY,
        'cross_day_soc': 'one continuous annual state trajectory',
        'purchase_Q_fixed': True, 'original_purchase_cost_paid_in_full': True,
        'emergency_charging': False, 'discharge_into_surplus': False,
        'simultaneous_charge_discharge': False,
        'actual_load_and_PV_seen_with_perfect_foresight': True,
        'relaxed_constraints': ['deadband', 'hard ramp', 'direction reversal limits'],
        'terminal_soc': 'only physical lower/upper bounds; no terminal credit',
        'oracle_action_traces_exported': False, 'eligible_as_causal_strategy': False,
        'included_in_dispatch_index': False, 'source_code_sha256': hashes,
        'raw_input_sha256': {name: digest(ROOT / 'data/raw' / name) for name in
                            ['附件1.csv', '附件2_小区负载.csv', '附件2_光伏发电实际功率.csv']},
        'target_8pct_yuan': .92 * BASELINE_COST,
        'protected_original_oracle': str(protected), 'protected_original_oracle_sha256': protection}
    save(OUT / 'protocol.json', protocol)
    bound = solve_fixed_purchase(SOURCE, OUT, .8)
    assert digest(SOURCE) == protocol['source_sha256']
    assert digest(protected) == protection
    threshold = .92 * BASELINE_COST
    # Use the dual bound to rule out success; a primal below threshold alone
    # does not establish a causal low-switch executor exists.
    excluded = bound['total_dual_bound_yuan'] > threshold
    summary = {'complete_diagnostic': True, 'eligible_as_causal_strategy': False,
        'oracle_uses_future_actual': True, 'oracle_action_traces_exported': False,
        'included_in_dispatch_index': False, 'protected_original_oracle_unchanged': True,
        'source_purchase_archive_unchanged': True, 'target_8pct_yuan': threshold,
        'frozen_purchase_cost_yuan': bound['frozen_planned_cost_yuan'],
        'minimum_emergency_primal_yuan': bound['minimum_emergency_cost_yuan'],
        'minimum_emergency_dual_yuan': bound['emergency_dual_objective_yuan'],
        'minimum_total_primal_yuan': bound['minimum_total_cost_yuan'],
        'minimum_total_dual_yuan': bound['total_dual_bound_yuan'],
        'primal_dual_gap_yuan': bound['primal_dual_gap_yuan'],
        'source_actual_total_yuan': bound['candidate_actual_total_cost_yuan'],
        'maximum_execution_only_saving_yuan': bound['maximum_saving_by_execution_only_yuan'],
        'dual_bound_above_target_yuan': bound['total_dual_bound_yuan'] - threshold,
        'execution_only_8pct_ruled_out': excluded,
        'conclusion': ('Even perfect-future physically valid battery execution of this exact Q cannot reach the 8% cost target; changing Q is necessary.'
                       if excluded else 'The physical fixed-Q lower bound does not rule out the 8% target, but does not establish a causal low-switch executor attaining it.'),
        'no_formal_forecast_or_controller_changed': True}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
