"""Fixed two-start historical-loss selection for five conditional daily pairs.

Each day's starting SOC/mask is the signed memory physical reference. The
alternative does not form one continuous multi-day policy. Selection sees
only available historical paths; actuals enter afterward for evaluation.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import optimize,objective
from experiments.exp008.controller_candidate import Data,plan_inventory
from experiments.exp008.planner import execute,ETA
from experiments.exp008.verify import verify_npz

OUT=Path('data/results/exp008/multistart_refinement')
SOURCE=Path('data/results/exp008/mode_planning_physical/load_memory_half_physical3_30days')
SUPPORT=Path('data/results/exp008/load_energy_memory/lp_bridge/memory_tree28_q08_buffer500_334days/supports.npz')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg={'days':[31,32,33,34,35],'fixed_starts':['archived_three_path_MIP','tree_quantile_0.8_LP'],
         'selection':'minimum same all-28-history closed_loop objective, before reading actual',
         'same_masks_and_starting_soc':True,'maxiter':120,'variation':0.,'wear':.002,'terminal':.45,
         'role':'five conditional daily interventions, not continuous annual replay',
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'support_sha256':hashlib.sha256(SUPPORT.read_bytes()).hexdigest(),
         'reference_sha256':hashlib.sha256((SOURCE/'dispatch.npz').read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
    with np.load(SUPPORT) as archive:
        supports=archive['supports'].copy()
    data=Data();rows=[]
    with np.load(SOURCE/'dispatch.npz') as archive:
        reference_q=archive['original'][:5].copy()
    for i,day in enumerate(cfg['days']):
        with np.load(SOURCE/f'planning_day{day}.npz') as archive:
            source={key:archive[key].copy() for key in archive.files}
        soc=float(source['initial_soc']);mode=int(source['initial_mode'])
        options={'charge_mask':source['allowed_charge'],'throughput':.002,
                 'variation':0.,'terminal':.45,'maxiter':120,'deadband':0.}
        baseline=optimize(source['purchase'],source['all_net_paths'],source['price'],soc,**options)
        parity=float(np.max(np.abs(baseline['purchase']-reference_q[i])))
        assert parity<1e-7,parity
        seed=plan_inventory(supports[i],source['price'],soc,{'quantile':.8,'wear':.002},final=False)
        second=optimize(seed['purchase'],source['all_net_paths'],source['price'],soc,**options)
        # Nothing from the target day's realized load/PV has been read above.
        selected=min((('original',baseline),('second',second)),key=lambda row:row[1]['metadata']['objective'])[0]
        observed=data.actual[day*144:(day+1)*144]
        results={}
        for label,solved in [('original',baseline),('second',second)]:
            q=solved['purchase']
            detail=execute(q,observed,source['price'],soc,charge_deadband=0.,charge_mask=source['allowed_charge'])
            detail.update(original=q,final=q.copy(),actual=observed.copy(),price=source['price'].copy())
            detail['fees']=np.stack((q*source['price'],np.zeros(144),np.zeros(144),
                                     5*detail['emergency']*source['price']),axis=-1)
            arrays={key:value[None,...] for key,value in detail.items()};arrays['days']=np.array([day])
            path=OUT/f'day{day}_{label}_conditional.npz'
            np.savez_compressed(path,**arrays)
            check=verify_npz(path,expected_days=1,start_day=day,initial_soc=soc,initial_mode=mode)
            assert check['passed'],check['errors']
            (OUT/f'day{day}_{label}_verification.json').write_text(json.dumps(check,indent=2))
            results[label]={'cost':float(detail['fees'].sum()),'final_soc':float(detail['states'][-1]),
                            'history_objective':solved['metadata']['objective'],
                            'solver':solved['metadata']}
        chosen=results[selected];old=results['original']
        depletion=max(0.,old['final_soc']-chosen['final_soc'])*ETA*5*data.fixed_price.max()
        rows.append({'day':day,'selected':selected,'reference_q_reproduction_error':parity,
            'original_cost':old['cost'],'selected_cost':chosen['cost'],
            'cost_change':chosen['cost']-old['cost'],'depletion_upper_bound_cost':depletion,
            'conservative_cost_change':chosen['cost']-old['cost']+depletion,
            'historical_objective_change':chosen['history_objective']-old['history_objective'],
            'original_final_soc':old['final_soc'],'selected_final_soc':chosen['final_soc']})
        (OUT/f'day{day}_optimization.json').write_text(json.dumps({'selected':selected,'results':results},indent=2))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'paired.csv',index=False)
    summary={'config':cfg,'selected_second_days':int((frame.selected=='second').sum()),
             'sum':frame.drop(columns=['day','selected']).sum().to_dict(),
             'goal_eligible':False,'no_actual_used_for_start_selection':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
