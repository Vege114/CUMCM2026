"""Single fixed annual continuation of the matched 30-day joint CNN policy."""

import hashlib
import json
import platform
from pathlib import Path

import scipy

from experiments.exp008.mode_planning_physical import OUT, run
from experiments.exp008.neural_joint_dispatch import JointForecasts


def main():
    name='joint_ridge28_refined_s3_334days'
    directory=OUT/name
    directory.mkdir(parents=True,exist_ok=True)
    caller={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform(),
            'scipy':scipy.__version__,'days':334,'scenarios':3,
            'forecast':'joint_shared_CNN_same_Ridge28',
            'fixed_hyperparameters':'identical_to_joint_ridge28_refined_s3_30days',
            'first_30_days':'reuse_each_original_array_and_continue_its_SOC_and_mode',
            'monthly_performance_early_stopping':False,
            'annual_comparison_scope':'exp006_and_existing_annual_methods; no pure-network annual ablation',
            'development_year_not_independent_test':True}
    (directory/'annual_protocol.json').write_text(json.dumps(caller,indent=2))
    (directory/'annual_caller_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return run(days=334,scenarios=3,forecast_override=JointForecasts(),case_name=name,
               prefix_directory=OUT/'joint_ridge28_refined_s3_30days')


if __name__=='__main__':
    main()
