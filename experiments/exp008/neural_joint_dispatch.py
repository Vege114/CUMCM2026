"""One matched physical-planner linkage for the joint CNN plus fixed Ridge28."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.exp008.forecast import Forecasts
from experiments.exp008.mode_planning_physical import OUT as PHYSICAL_OUT
from experiments.exp008.mode_planning_physical import run as physical_run
from experiments.exp008.neural_joint_calibration import OUT as CALIBRATION_OUT
from experiments.exp008.neural_joint_calibration import JointStore

OUT=Path('data/results/exp008/neural_joint_dispatch')


class JointForecasts(Forecasts):
    def __init__(self):
        super().__init__()
        self.store=JointStore(calibrated=True)
        self.calibration='joint_ridge28'

    def get(self,day,slot=0,scenario='2'):
        result=super().get(day,slot,scenario)
        result['audit'].update(
            base_forecast='exp008_joint_shared_cnn' if day>=31 else 'periodic_cold_start',
            output_calibration='same_fixed_Ridge28' if day>=31 else None,
            base_architecture_unchanged=day<31,
            joint_training='monthly_from_scratch_prior7day_validation_seed42_CPU' if day>=31 else None,
            calibration_labels_end_exclusive=day*144 if day>=31 else None)
        return result


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'days':30,'forecast':'joint_network_then_unchanged_Ridge28',
              'comparison_forecast':'original_network_then_unchanged_Ridge28',
              'planner':'same_physical_shared_hour_mode_scenario3',
              'full_28_complete_history_paths_for_greedy_refinement':True,
              'scenario_error_forecast_source_matches_current_forecast':True,
              'switching':50.,'mip_seconds':5.,'refinement_maxiter':120,
              'no_deadband':True,'each_policy_continuous_soc':True,
              'only_forecast_pipeline_changed':True,'annual_claim':False,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'joint_calibrated_archive_sha256':hashlib.sha256((CALIBRATION_OUT/'joint_ridge28.npz').read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    name='joint_ridge28_refined_s3_30days'
    result=physical_run(days=30,scenarios=3,checkpoint_days=5,
                        forecast_override=JointForecasts(),case_name=name)
    if not (PHYSICAL_OUT/name/'summary.json').exists():
        return result
    reference=json.loads((PHYSICAL_OUT/'refined_s3_30days/summary.json').read_text())
    rows=[{'name':label,'total_cost':entry['total_cost'],**entry['battery']}
          for label,entry in [('original_ridge28',reference),('joint_ridge28',result)]]
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    summary={'days':30,'original_cost':reference['total_cost'],'joint_cost':result['total_cost'],
             'cost_change_yuan':result['total_cost']-reference['total_cost'],
             'relative_change_pct':100*(result['total_cost']/reference['total_cost']-1),
             'original_battery':reference['battery'],'joint_battery':result['battery'],
             'both_verified':result['verification']['passed'] and reference['verification']['passed'],
             'same_planner_hyperparameters':True,'current_and_historical_forecasts_both_changed':True,
             'annual_goal_claimed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    run()
