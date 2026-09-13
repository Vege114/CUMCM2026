"""One fixed full-year Q3/Q4-3 cost-only treatment, preserving exp008 source.

The absolute case path routes the unchanged run_case into exp009. Only the
two auxiliary objective coefficients and the charging deadband become zero.
"""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import scipy

from experiments.exp008.absolute_hgb_update_dispatch import override
from experiments.exp008.forecast_absolute_hgb import OUT as FORECAST_OUT, PRIMARY
from experiments.exp008.issued_residual_paths import issued_error_paths
from experiments.exp008.planner import Settings
from experiments.exp008.run import OUT as ORIGINAL_RUN_OUT, initial_state, run_case
# Preload run_case's lazy local import before freezing the actual dependency closure.
from experiments.exp008.unified_forecast import UnifiedForecasts  # noqa: F401
from experiments.exp009.update_cost_only_audit import OUT, BASE, ROOT, audit_scenario, digest, save

SETTINGS = Settings(throughput_penalty=0., variation_penalty=0., future_shortfall_weight=1.5)
SCENARIOS = ('3', '4-3')


def source_closure():
    paths = set()
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'experiments'):
                paths.add(path)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def original_artifacts():
    paths = set()
    for scenario in SCENARIOS:
        paths.update(p for p in (BASE / scenario).iterdir() if p.is_file())
        paths.add(ROOT / f'data/results/exp002/warmup_{scenario}.npz')
    paths.update(p for p in FORECAST_OUT.iterdir() if p.is_file())
    paths.update((FORECAST_OUT / 'models').glob('*.joblib'))
    paths.update((ROOT / 'data/raw').glob('*.csv'))
    paths.update(ROOT / 'data/results/exp004' / name for name in
                 ('prediction_manifest.json', 'predictions.npz'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def check_frozen(protocol):
    assert digest(ROOT / 'experiments/exp009/protocol.json') == protocol['root_protocol_sha256']
    assert source_closure() == protocol['imported_local_source_sha256'], 'local dependency closure changed'
    for key, expected in protocol['original_artifact_sha256'].items():
        assert digest(ROOT / key) == expected, ('original artifact changed', key)
    return {'passed': True, 'imported_local_sources': len(protocol['imported_local_source_sha256']),
            'original_artifacts': len(protocol['original_artifact_sha256']),
            'root_protocol_sha256': protocol['root_protocol_sha256']}


def future_checks():
    rows = []
    for scenario in SCENARIOS:
        for day, slot in ((31, 0), (32, 36), (60, 72), (180, 108)):
            before = override()
            changed = copy.copy(before.data)
            cutoff = day * 144 + slot
            changed.actual = before.data.actual.copy()
            changed.actual[cutoff:] += [100000., 50000., 1000.]
            changed._forecasts = {k: np.asarray(v).copy() + (90000. if k > cutoff else 0.)
                                  for k, v in before.data.forecasts.items()}
            after = override(changed)
            a, b = before.get(day, slot, scenario), after.get(day, slot, scenario)
            ra = issued_error_paths(before, day, slot, scenario=scenario)
            rb = issued_error_paths(after, day, slot, scenario=scenario)
            for key in ('load_kw', 'pv_kw', 'price'):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ('errors_kw', 'errors_kwh', 'price_errors', 'origins', 'label_stops_exclusive'):
                np.testing.assert_array_equal(ra[key], rb[key])
            assert not b['audit']['known_future_price']
            assert ra['label_stops_exclusive'].max() <= day * 144
            rows.append({'scenario': scenario, 'day': day, 'slot': slot,
                         'future_actual_and_unreleased_PV_mutation_identical': True,
                         'same_issue_history_identical': True})
    save(OUT / 'causality_verification.json', {
        'passed': True, 'checks': rows,
        'raw_HGB_and_Ridge_memory_training_causality': str(FORECAST_OUT / 'causality_verification.json'),
        'no_retraining': True,
        'information_boundary': 'Full CSVs are loaded by Data; causality refers to legal prefix use for planning and is tested by future suffix mutation, not a claim that later rows were never loaded.'})


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing experiment evidence: {OUT}')
    root_protocol_path = ROOT / 'experiments/exp009/protocol.json'
    root_protocol = json.loads(root_protocol_path.read_text())
    assert root_protocol['formal_run_authorized'] and root_protocol['candidate_count_per_question'] == 1
    assert root_protocol['evaluation']['days'] == 334
    for scenario in SCENARIOS:
        treatment = root_protocol['treatment'][scenario]
        assert treatment['throughput_penalty'] == treatment['variation_penalty'] == treatment['charge_deadband_kwh'] == 0
        assert treatment['representative_paths'] == 7 and treatment['future_shortfall_weight'] == 1.5
        assert treatment['updates'] == [0, 36, 72, 108]
    # Construct store-backed adapters only; no fitting or candidate search occurs.
    prepared = {scenario: override() for scenario in SCENARIOS}
    states = {scenario: initial_state(scenario) for scenario in SCENARIOS}
    for scenario, (soc, power) in states.items():
        assert soc == root_protocol['fixed']['initial_soc_kwh'][scenario]
        assert power == root_protocol['fixed']['annual_initial_power_kw']
        assert int(np.sign(power)) == root_protocol['fixed']['annual_initial_mode']
    case = str((OUT / 'full334').resolve())
    assert ORIGINAL_RUN_OUT / case / '3' == OUT / 'full334' / '3'
    assert not (OUT / 'full334').is_relative_to(ORIGINAL_RUN_OUT)
    models = list((FORECAST_OUT / 'models').glob('*.joblib'))
    assert len(models) == 22
    protocol = {
        'experiment_id': 'exp009', 'scope': ['3', '4-3'], 'days': 334,
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'root_protocol_sha256': digest(root_protocol_path),
        'root_protocol': root_protocol,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'settings': asdict(SETTINGS), 'deadband_kwh': 0., 'seed': 42,
        'initial_states_soc_power': states, 'forecast_identity': prepared['3'].forecast_identity(),
        'model_files_count': len(models), 'new_fits': 0,
        'imported_local_source_sha256': source_closure(),
        'original_artifact_sha256': original_artifacts(),
        'runtime': {'python': sys.version, 'platform': platform.platform(),
                    'numpy': np.__version__, 'scipy': scipy.__version__},
        'output_case_absolute': case, 'baseline': str(BASE),
        'predeclared_no_performance_gate': True, 'no_candidate_selection': True,
        'terminal_economic_proxy_retained': True,
    }
    OUT.mkdir(parents=True)
    save(OUT / 'run_protocol.json', protocol)
    shutil.copy2(root_protocol_path, OUT / 'root_protocol.json')
    for relative in protocol['imported_local_source_sha256']:
        destination = OUT / 'source_archive' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    began = perf_counter()
    try:
        future_checks()
        check_frozen(protocol)
        annual = []
        for scenario in SCENARIOS:
            run_case(case, scenario, method='joint', settings=SETTINGS, days=334,
                     deadband=0., updates=True, issued_residuals=True,
                     forecast_override=prepared[scenario])
            result = audit_scenario(scenario)
            save(OUT / f'preservation_after_{scenario}.json', check_frozen(protocol))
            annual.append(result)
            print('INDEPENDENTLY_VERIFIED', scenario, result['candidate']['recomputed_total_cost'], flush=True)
        preservation = check_frozen(protocol)
        save(OUT / 'preservation_after.json', preservation)
        save(OUT / 'summary.json', {'complete': True, 'passed': True, 'annual': annual,
                                  'preservation': preservation, 'total_orchestration_wall_seconds': perf_counter() - began,
                                  'new_forecast_fits': 0, 'selection_performed': False,
                                  'development_year_not_independent_test': True})
    except BaseException as exc:
        save(OUT / 'failure.json', {'complete': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
                                  'elapsed_seconds': perf_counter() - began,
                                  'policy': 'Preserve partial evidence; no automatic budget increase, retuning or second candidate.'})
        raise


if __name__ == '__main__':
    main()
