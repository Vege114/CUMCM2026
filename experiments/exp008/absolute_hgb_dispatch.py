"""Predeclared full-year direct-HGB pipeline with unchanged physical dispatch."""

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy
import sklearn

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.forecast_absolute_hgb import OUT as FORECAST_OUT
from experiments.exp008.forecast_absolute_hgb import PRIMARY
from experiments.exp008.mode_planning_physical import OUT, run

ROOT=Path(__file__).resolve().parents[2]
STUDY=OUT/'direct_hgb_fixed_physical3_evaluation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(days):
    directory=OUT/f'direct_hgb_memory_physical3_{days}days'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        raise FileExistsError('Completed evaluation remains immutable')
    forecast=AbsoluteHGBForecasts()
    dependencies=['experiments/exp008/'+name+'.py' for name in (
        'absolute_hgb_dispatch','absolute_hgb_dispatch_audit','absolute_hgb_forecast_adapter',
        'forecast_absolute_hgb','forecast_net_hgb','forecast_override_adapter','forecast_shape_diagnostic',
        'load_energy_memory','mode_planning_physical','closed_loop','planner','forecast',
        'forecast_calibration','neural_joint_calibration','controller_candidate','unified_forecast','verify')]
    dependencies += ['experiments/common/neural_v2/physics.py','experiments/common/neural_v2/data.py',
                     'experiments/problem2/exp003/data.py','experiments/problem2/exp004/data.py',
                     'experiments/problem2/exp004/predict.py','experiments/problem2/exp004/protocol.json']
    dependencies += [name for name in ('pyproject.toml','uv.lock') if (ROOT/name).exists()]
    source_hashes={}
    for name in dependencies:
        source=ROOT/name;destination=directory/'source_archive'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);source_hashes[name]=digest(source)
    artifacts=[path for path in FORECAST_OUT.iterdir() if path.is_file()]
    artifacts+=sorted((FORECAST_OUT/'models').glob('*.joblib'))
    artifact_hashes={}
    for source in artifacts:
        name=str(source.relative_to(FORECAST_OUT));destination=directory/'forecast_archive'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);artifact_hashes[name]=digest(source)
    with np.load(FORECAST_OUT/f'{PRIMARY}.npz') as archive:
        np.testing.assert_array_equal(forecast.store.values,archive['values'])
        np.testing.assert_array_equal(forecast.store.origins,archive['origins'])
    np.savez_compressed(directory/'issued_forecasts.npz',origins=forecast.store.origins,values=forecast.store.values)
    provenance={'primary':PRIMARY,'interpretation':'complete direct absolute HGB + same Ridge28 + same fixed0.5 memory pipeline',
                'raw_HGB_alone_not_claimed_superior':True,'days':days,
                'source_data_sha256':forecast.data.hashes,'source_sha256':source_hashes,
                'forecast_artifact_sha256':artifact_hashes,'issued_archive_sha256':digest(directory/'issued_forecasts.npz'),
                'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
                'python':platform.python_version(),'platform':platform.platform(),
                'scenarios':3,'block_slots':6,'switching':50.,'mip_seconds':5.,'gap':.002,
                'all28_historical_paths_for_refinement':True,'refinement_maxiter':120,
                'wear':.002,'variation':0.,'deadband':0.,'hyperparameters_changed':False,
                'predeclared_full_evaluation_days':334,'short_window_cost_selection_gate':False,
                'development_not_independent_test':True}
    (directory/'complete_provenance.json').write_text(json.dumps(provenance,indent=2))
    shutil.copy2(STUDY/'evaluation_protocol.json',directory/'evaluation_protocol.json')
    return directory,forecast


def main():
    from experiments.exp008.absolute_hgb_dispatch_audit import main as audit
    STUDY.mkdir(parents=True,exist_ok=True)
    protocol={'evaluation_days':334,'pipeline':PRIMARY,'fixed_candidate_count':1,
              'comparison':'completed Joint_Ridge28_memory_half physical3 full334day method',
              'first3days':'computational feasibility, physical constraints, sources and causal history only',
              'first3day_cost_is_not_a_selection_gate':True,
              'continue_after_feasibility':'same first3 arrays plus same continuous SOC and mode for next331days',
              'full_year_selection_rule':'same exp0068pct cost gate + fewer nonidle reversals + hard physics',
              'stop_from_partial_month_cost':False,'mode_block_slots':6,
              'no_new_training_or_hyperparameter_changes':True,
              'complete_pipeline_comparison_not_raw_HGB_superiority':True,
              'development_year_not_independent_test':True,'source_sha256':digest(__file__)}
    path=STUDY/'evaluation_protocol.json'
    if path.exists():
        raise FileExistsError('This fixed annual evaluation has already been started')
    path.write_text(json.dumps(protocol,indent=2))
    directory,forecast=prepare(3)
    run(days=3,scenarios=3,forecast_override=forecast,case_name=directory.name)
    checked=audit(3)
    if not checked['independent_verification_passed']:
        raise AssertionError('First3day computational or physical audit failed')
    (STUDY/'feasibility_check.json').write_text(json.dumps({
        'passed':True,'days':3,'cost_not_used_for_decision':True,
        'audit_sha256':digest(directory/'independent_verification.json')},indent=2))
    print('FIRST3_FEASIBILITY_PASSED; CONTINUE_FIXED334',flush=True)
    directory,forecast=prepare(334)
    run(days=334,scenarios=3,forecast_override=forecast,case_name=directory.name,
        prefix_directory=OUT/'direct_hgb_memory_physical3_3days')
    return audit(334)


if __name__=='__main__':
    main()
