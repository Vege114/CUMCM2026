"""Bounded three-setting purchase replanning after a changed forecast model.

The middle control exactly reproduces the signed direct-HGB LP bridge.
Only its numpy quantile parameter changes, not the forecasts or risk samples.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from experiments.exp008.controller_candidate import Data,INITIAL_SOC,plan_inventory,execute_inventory
from experiments.exp008.verify import verify_npz
from experiments.exp008.risk_window import SPEC

OUT=Path('data/results/exp008/hgb_quantile_replan')
REFERENCE=Path('data/results/exp008/forecast_absolute_hgb/lp_bridge/'
               'direct_hgb_ridge28_memory_tree28_q08_buffer500_334days')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'fixed_parameters':[.75,.8,.85],'control':.8,'days':334,
              'reference':str(REFERENCE),'risk_reused_because_independent_of_battery_policy':True,
              'only_quantile_changed':True,'fixed_other_spec':SPEC,
              'no_new_configuration_added_after_results':True,
              'evaluation_role':'2025 development; not independent model selection',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_hashes':{name:hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()
                    for name in ('supports.npz','audit.json','dispatch_2.npz')}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with np.load(REFERENCE/'supports.npz') as archive:
        supports=archive['supports'].copy()
    audits=json.loads((REFERENCE/'audit.json').read_text())
    data=Data();summaries=[]
    for quantile in protocol['fixed_parameters']:
        directory=OUT/f'q{quantile:.2f}_334days';directory.mkdir(exist_ok=True)
        if (directory/'summary.json').exists():
            raise FileExistsError('Completed quantile replay is immutable')
        spec={**SPEC,'calibration':'direct_hgb_ridge28_memory_half','quantile':quantile}
        soc,mode=INITIAL_SOC,1;details=[];rows=[]
        for i,day in enumerate(range(31,365)):
            plan=plan_inventory(supports[i],data.fixed_price,soc,spec,final=day==364)
            actual=data.actual[day*144:(day+1)*144]
            detail,mode=execute_inventory(plan,actual,data.fixed_price,soc,mode,spec)
            details.append(detail);soc=float(detail['states'][-1])
            rows.append({'day':day,'total_cost':float(detail['fees'].sum()),
                         'planned_cost':float(detail['fees'][:,0].sum()),
                         'emergency_cost':float(detail['fees'][:,3].sum()),'final_soc':soc})
        arrays={key:np.stack([d[key] for d in details]) for key in details[0]}
        arrays['days']=np.arange(31,365)
        if quantile==.8:
            with np.load(REFERENCE/'dispatch_2.npz') as reference:
                parity={key:float(np.max(np.abs(arrays[key]-reference[key]))) for key in arrays}
            if any(value>1e-7 for value in parity.values()):
                raise AssertionError(parity)
            (directory/'control_parity.json').write_text(json.dumps(parity,indent=2))
        np.savez_compressed(directory/'dispatch_2.npz',**arrays)
        (directory/'audit.json').write_text(json.dumps(audits,indent=2))
        check=verify_npz(directory/'dispatch_2.npz',expected_days=334,audit_path=directory/'audit.json')
        assert check['passed'],check['errors']
        summary={'spec':spec,'total_cost':check['billing']['total_cost'],
                 'billing':check['billing'],'battery':check['battery_metrics'],
                 'verification':check,'only_quantile_changed':True}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2))
        pd.DataFrame(rows).to_csv(directory/'daily.csv',index=False)
        summaries.append({'quantile':quantile,**check['billing'],**check['battery_metrics']})
        print('HGB q',quantile,'cost',summary['total_cost'],'reversals',summary['battery']['direction_reversals'],flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'comparison.csv',index=False)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    run()
