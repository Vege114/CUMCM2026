"""Matched numerical-convergence diagnostic of a fixed shared-mode MIP plan.

One MIP per day supplies identical initial purchase, modes, paths and SOC to
L-BFGS-B with iteration limits 120 and 600. Reference SOC follows the 120 arm.
Thus 600's daily outcomes are conditional paired interventions, not a claimed
continuous alternate rollout or an eligible full-year objective result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import objective, optimize
from experiments.exp008.controller_candidate import INITIAL_SOC, Data
from experiments.exp008.mode_planning import plan
from experiments.exp008.planner import execute
from experiments.exp008.unified_forecast import UnifiedForecasts
from experiments.exp008.verify import battery_metrics, verify_npz

OUT = Path('data/results/exp008/mode_refinement_diagnostic')


def run(days=5):
    directory = OUT / f'paired_{days}days'
    if (directory/'summary.json').exists():
        raise FileExistsError('Completed paired diagnostics are immutable')
    directory.mkdir(parents=True,exist_ok=True)
    cfg = {'days':days,'calibration':'ridge_28','mip_scenarios':7,
           'refinement_history_days':28,'block_slots':6,'switching':50.,
           'mip_seconds':5.,'mip_gap':.002,'wear':.002,'variation':0.,
           'terminal':.45,'deadband':0.,'maxiter_arms':[120,600],
           'only_changed_parameter':'L-BFGS-B maximum iterations',
           'paired_same_purchase_initialization_mode_paths_soc_prices':True,
           'day_start_soc_follows_120_reference':True,
           'alternative_600_is_conditional_daily_diagnostic_not_continuous_rollout':True,
           'development_on_2025_not_independent_validation':True}
    (directory/'protocol.json').write_text(json.dumps(cfg,indent=2))
    for module in ('mode_refinement_diagnostic','mode_planning','closed_loop'):
        source=Path(f'experiments/exp008/{module}.py')
        (directory/f'{module}_snapshot.py').write_bytes(source.read_bytes())
    data,forecast=Data(),UnifiedForecasts(calibration=cfg['calibration'])
    soc,mode=INITIAL_SOC,1
    rows,audits,reference=[],[],[]
    began=perf_counter()
    for day in range(31,31+days):
        daily=directory/f'day{day:03d}'
        daily.mkdir(exist_ok=True)
        issue=forecast.get(day,scenario='2')
        errors=forecast.net_error_paths(day,'2',limit=28)
        net=(issue['load_kw']-issue['pv_kw'])/6
        paths=net[None,:]+errors['errors_kwh']
        selected=np.linspace(0,len(paths)-1,min(7,len(paths))).astype(int)
        initial=plan(paths[selected],data.fixed_price,soc,mode,
                     block_slots=6,switching=50.,wear=.002,seconds=5.,gap=.002,
                     final=day==364)
        options={'charge_mask':initial['allowed_charge'],'throughput':.002,
                 'variation':0.,'terminal':0. if day==364 else .45,
                 'deadband':0.,'emergency_weight':5.}
        starting_objective=objective(initial['purchase'],paths,data.fixed_price,soc,**options)[0]
        np.savez_compressed(daily/'shared_inputs.npz',purchase_initial=initial['purchase'],
            allowed_charge=initial['allowed_charge'],all_historical_net_paths=paths,
            selected_mip_path_indices=selected,price=data.fixed_price,
            initial_soc=np.array(soc),initial_mode=np.array(mode))
        metadata={'day':day,'initial_soc':soc,'initial_mode':mode,
                  'forecast':issue['audit'],'history':errors['audit'],
                  'mip':initial['metadata'],'initial_training_objective':starting_objective,
                  'shared_input_sha256':hashlib.sha256((daily/'shared_inputs.npz').read_bytes()).hexdigest()}
        actual=data.actual[day*144:(day+1)*144]
        outcomes={}
        for maxiter in (120,600):
            solved=optimize(initial['purchase'].copy(),paths,data.fixed_price,soc,
                            maxiter=maxiter,**options)
            q=solved['purchase']
            detail=execute(q,actual,data.fixed_price,soc,
                           charge_deadband=0.,charge_mask=initial['allowed_charge'])
            detail.update(original=q,final=q.copy(),actual=actual.copy(),price=data.fixed_price.copy())
            detail['fees']=np.stack((q*data.fixed_price,np.zeros(144),np.zeros(144),
                                    5*detail['emergency']*data.fixed_price),axis=-1)
            archive={key:value[None,...] for key,value in detail.items()}
            archive['days']=np.array([day])
            archive_path=daily/f'arm{maxiter}.npz'
            np.savez_compressed(archive_path,**archive)
            check=verify_npz(archive_path,expected_days=1,start_day=day,
                             initial_soc=soc,initial_mode=mode)
            if not check['passed']:
                raise AssertionError(check['errors'])
            (daily/f'arm{maxiter}_verification.json').write_text(json.dumps(check,indent=2))
            row={'day':day,'maxiter':maxiter,'initial_soc':soc,
                 'initial_training_objective':starting_objective,
                 'training_objective':solved['metadata']['objective'],
                 'training_gain_from_mip':starting_objective-solved['metadata']['objective'],
                 'actual_cost':float(detail['fees'].sum()),
                 'planned_cost':float(detail['fees'][:,0].sum()),
                 'emergency_cost':float(detail['fees'][:,3].sum()),
                 **solved['metadata'],**battery_metrics(archive,initial_mode=mode)}
            rows.append(row)
            outcomes[maxiter]=detail
            metadata[f'arm{maxiter}']=solved['metadata']
        (daily/'audit.json').write_text(json.dumps(metadata,indent=2))
        audits.append(metadata)
        reference.append(outcomes[120])
        soc=float(outcomes[120]['states'][-1])
        signs=np.sign(outcomes[120]['charge']-outcomes[120]['discharge'])
        nonzero=signs[signs!=0]
        if len(nonzero):
            mode=int(nonzero[-1])
        print(f'paired {day-30}/{days} elapsed {perf_counter()-began:.1f}s',flush=True)
    ref={key:np.stack([detail[key] for detail in reference]) for key in reference[0]}
    ref['days']=np.arange(31,31+days)
    np.savez_compressed(directory/'reference120_continuous.npz',**ref)
    check=verify_npz(directory/'reference120_continuous.npz',expected_days=days)
    assert check['passed'],check['errors']
    (directory/'reference120_verification.json').write_text(json.dumps(check,indent=2))
    frame=pd.DataFrame(rows)
    frame.to_csv(directory/'daily_arms.csv',index=False)
    a,b=(frame[frame.maxiter==limit].set_index('day') for limit in (120,600))
    pairs=pd.DataFrame({'day':a.index,
        'training_objective_delta_600_minus_120':b.training_objective-a.training_objective,
        'actual_cost_delta_600_minus_120':b.actual_cost-a.actual_cost,
        'planned_cost_delta_600_minus_120':b.planned_cost-a.planned_cost,
        'emergency_cost_delta_600_minus_120':b.emergency_cost-a.emergency_cost,
        'reference_training_objective':a.training_objective,
        'reference_actual_cost':a.actual_cost})
    pairs.to_csv(directory/'paired_deltas.csv',index=False)
    summary={'protocol':cfg,'continuous_120_cost':float(a.actual_cost.sum()),
        'conditional_600_cost_sum_not_a_rollout':float(b.actual_cost.sum()),
        'paired_actual_cost_delta':float((b.actual_cost-a.actual_cost).sum()),
        'paired_training_objective_delta':float((b.training_objective-a.training_objective).sum()),
        'paired_training_relative_change_pct':float(100*(b.training_objective.sum()/a.training_objective.sum()-1)),
        'converged_120_days':int(a.success.sum()),'converged_600_days':int(b.success.sum()),
        'iteration_counts_120':a.iterations.tolist(),'iteration_counts_600':b.iterations.tolist(),
        'all_daily_arms_physically_verified':True,'reference_120_continuously_verified':True,
        'future_actual_not_used_for_purchases':True,'elapsed_seconds':perf_counter()-began}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--days',type=int,default=5)
    args=parser.parse_args()
    run(args.days)
