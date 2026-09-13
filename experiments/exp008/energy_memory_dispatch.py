"""Fixed gain 0.5 load-memory linkage with the unchanged physical mode policy."""

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy

from experiments.exp008.load_energy_memory import EnergyMemoryForecasts
from experiments.exp008.mode_planning_physical import OUT, run

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(days=30,amended_evaluation=False):
    if days not in (30,334):
        raise ValueError('Only the predeclared 30-day linkage and annual continuation are supported')
    case_name=f'load_memory_half_physical3_{days}days'
    directory=OUT/case_name
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        raise FileExistsError('Completed experiment is immutable')
    if days==334:
        gate=json.loads((OUT/'load_memory_half_physical3_30days/paired_findings.json').read_text())
        if not gate['annual_extension_gate_passed'] and not amended_evaluation:
            raise AssertionError('Predeclared pilot gate did not pass')
        if amended_evaluation:
            evidence=ROOT/'data/results/exp008/load_memory_scale_diagnostic/findings.json'
            diagnostics=json.loads(evidence.read_text())
            amendment={'evaluation_days':334,'original_30day_extension_gate_passed':gate['annual_extension_gate_passed'],
                       'original_30day_gate_preserved_not_reclassified':True,
                       'decision':'separate full-year development evaluation authorized after new window-representativeness evidence',
                       'authorization':'root task explicitly requested this amended fixed-parameter annual run',
                       'evidence':{'first30_additive_daily_mean_sse_change_pct':diagnostics['first30_period_29_pairs']['additive_half_sse_change_vs_raw_pct'],
                                   'annual_additive_daily_mean_sse_change_pct':diagnostics['all333_pairs']['additive_half_sse_change_vs_raw_pct'],
                                   'completed_fixed_LP_joint_cost':13416463.33,
                                   'completed_fixed_LP_memory_cost':13254749.36,
                                   'diagnostic_artifact':str(evidence),'diagnostic_sha256':digest(evidence)},
                       'candidate_hyperparameters_changed':False,'gain':.5,
                       'prefix_30days':'reuse each verified array and continue same SOC and previous nonidle direction',
                       'rerun_30day_prefix':False,'stop_based_on_intermediate_months':False,
                       'final_acceptance':'same exp006334-day baseline; at least8pct cost reduction and fewer nonidle reversals plus physics',
                       'independent_unseen_test_claimed':False,
                       'original_gate_artifact_sha256':digest(OUT/'load_memory_half_physical3_30days/paired_findings.json')}
            (directory/'amended_evaluation_protocol.json').write_text(json.dumps(amendment,indent=2))
    forecast=EnergyMemoryForecasts()
    dependencies=['experiments/exp008/'+name+'.py' for name in (
        'energy_memory_dispatch','load_energy_memory','mode_planning_physical','closed_loop',
        'planner','forecast','forecast_calibration','neural_joint_calibration','neural_joint',
        'controller_candidate','unified_forecast','verify')]
    dependencies += ['experiments/common/neural_v2/physics.py','experiments/common/neural_v2/data.py',
                     'experiments/problem2/exp003/data.py','experiments/problem2/exp004/data.py',
                     'experiments/problem2/exp004/predict.py','experiments/problem2/exp004/protocol.json']
    dependencies += [name for name in ('pyproject.toml','uv.lock') if (ROOT/name).exists()]
    source_hashes={}
    for name in dependencies:
        source=ROOT/name;destination=directory/'source_archive'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        source_hashes[name]=digest(source)
    artifacts=['load_energy_memory/predictions.npz','load_energy_memory/protocol.json',
               'load_energy_memory/prediction_audit.json','neural_joint_calibration/joint_ridge28.npz',
               'neural_joint_calibration/joint_ridge28_audit.json',
               'neural_joint_calibration/manifest.json','neural_joint/predictions.npz',
               'neural_joint/manifest.json','neural_joint/protocol.json']
    manifest=json.loads((ROOT/'data/results/exp008/neural_joint/manifest.json').read_text())
    model_directory=Path(manifest['run_directory'])
    artifacts += [str(path.relative_to(ROOT/'data/results/exp008'))
                  for path in sorted(model_directory.glob('joint_m*_s42.*'))]
    artifact_hashes={}
    for name in artifacts:
        source=ROOT/'data/results/exp008'/name;destination=directory/'forecast_archive'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        artifact_hashes[name]=digest(source)
    with np.load(ROOT/'data/results/exp008/load_energy_memory/predictions.npz') as archive:
        np.testing.assert_array_equal(forecast.store.values,archive['values'])
        np.testing.assert_array_equal(forecast.store.origins,archive['origins'])
    np.savez_compressed(directory/'issued_memory_forecasts.npz',
                        origins=forecast.store.origins,values=forecast.store.values,
                        base_values=forecast.store.base_values)
    provenance={'gain':.5,'forecast_name':forecast.calibration,'days':days,'candidate_count':1,
                'hyperparameter_search':False,'current_and_risk_history_use_same_corrected_forecasts':True,
                'fixed_physical_policy':'3_equally_spaced_history_scenarios_hourly_modes_kappa50_5s_gap0.002',
                'fixed_refinement':'all28_complete_paths_same_greedy_wear0.002_TV0_deadband0_maxiter120',
                'initial_soc':1421.7991105135516,'source_data_sha256':forecast.data.hashes,
                'source_sha256':source_hashes,'artifact_sha256':artifact_hashes,
                'python':platform.python_version(),'platform':platform.platform(),
                'numpy':np.__version__,'scipy':scipy.__version__,
                'archive_sha256':digest(directory/'issued_memory_forecasts.npz'),
                'development_year_not_independent_test':True}
    (directory/'complete_provenance.json').write_text(json.dumps(provenance,indent=2))
    return run(days=days,scenarios=3,forecast_override=forecast,case_name=case_name,
               prefix_directory=OUT/'load_memory_half_physical3_30days' if days==334 else None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--days',type=int,default=30)
    parser.add_argument('--amended-evaluation',action='store_true')
    args=parser.parse_args();main(args.days,args.amended_evaluation)
