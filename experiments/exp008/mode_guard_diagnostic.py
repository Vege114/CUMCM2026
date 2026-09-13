"""Fixed-purchase diagnostic of a causal large-action hourly mode override.

Purchases retain their source policy's virtual SOC lineage; they are not
replanned for the alternate actual SOC. This is an execution diagnostic, not
a jointly aligned candidate or a goal-eligible result. No future observation
is inspected by execute_guard.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp008.planner import ETA,HIGH,LIMIT,LOW,execute
from experiments.exp008.verify import verify_npz

OUT=Path('data/results/exp008/mode_guard_diagnostic')
SOURCE=Path('data/results/exp008/mode_planning_physical/joint_ridge28_refined_s3_334days/dispatch.npz')


def execute_guard(q,actual,price,soc,mask,threshold=.3*LIMIT):
    count=len(q)
    c,d,e,w=(np.zeros(count) for _ in range(4))
    s=np.r_[soc,np.zeros(count)]
    modes=np.zeros(count,dtype=bool)
    overrides=0
    for t in range(count):
        if t%6==0:
            mode=bool(mask[t])
            changed_this_hour=False
        b=q[t]+(actual[t,1]-actual[t,0])/6
        positive=b>=0.
        possible=(min(b,LIMIT,max(0.,(HIGH-s[t])/ETA)) if positive
                  else min(-b,LIMIT,max(0.,(s[t]-LOW)*ETA)))
        if not changed_this_hour and mode!=positive and possible>=threshold:
            mode=positive
            changed_this_hour=True
            overrides+=1
        if mode==positive:
            if positive:
                c[t]=possible
            else:
                d[t]=possible
        w[t]=max(0.,b-c[t]);e[t]=max(0.,-b-d[t])
        s[t+1]=s[t]+ETA*c[t]-d[t]/ETA
        modes[t]=mode
    return dict(charge=c,discharge=d,emergency=e,surplus=w,states=s,
                executed_charge_mode=modes),overrides


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'threshold_kwh':.3*LIMIT,'minimum_potential_override_power_kw':1500.,
              'max_overrides_per_hour':1,'override_remains_until_hour_boundary':True,
              'candidate_count':1,'goal_eligible':False,
              'purchase_state_lineage':'source virtual SOC, not alternate actual SOC',
              'role':'continuous physical fixed-purchase execution diagnostic',
              'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with np.load(SOURCE) as original:
        source={key:original[key].copy() for key in original.files}
    results={}
    for method in ('original_mask','guard_1500kw','unrestricted_greedy'):
        soc=float(source['states'][0,0]);parts=[];overrides=0
        for i in range(len(source['days'])):
            q,actual,price,mask=(source[key][i] for key in ('original','actual','price','allowed_charge'))
            if method=='guard_1500kw':
                detail,count=execute_guard(q,actual,price,soc,mask);overrides+=count
            else:
                detail=execute(q,actual,price,soc,charge_deadband=0.,
                               charge_mask=mask if method=='original_mask' else None)
            detail.update(original=q,final=q.copy(),actual=actual,price=price)
            detail['fees']=np.stack((q*price,np.zeros(144),np.zeros(144),5*price*detail['emergency']),axis=-1)
            parts.append(detail);soc=float(detail['states'][-1])
        arrays={key:np.stack([part[key] for part in parts]) for key in parts[0]}
        arrays['days']=source['days']
        if method=='original_mask':
            parity={key:float(np.max(np.abs(arrays[key]-source[key])))
                    for key in arrays if key in source}
            assert all(value<1e-8 for value in parity.values()),parity
        path=OUT/f'{method}_counterfactual_trace.npz'
        np.savez_compressed(path,**arrays)
        check=verify_npz(path,expected_days=334)
        assert check['passed'],check['errors']
        check['eligible_for_goal_claim']=False
        (OUT/f'{method}_verification.json').write_text(json.dumps(check,indent=2))
        results[method]={'total_cost':check['billing']['total_cost'],
                         'battery':check['battery_metrics'],'mode_overrides':overrides}
    summary={'protocol':protocol,'results':results,
             'source_controller_reproduction_max_error':max(parity.values())}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({key:{'cost':value['total_cost'],'reversals':value['battery']['direction_reversals'],
                    'final_soc':value['battery']['final_soc'],'overrides':value['mode_overrides']}
                    for key,value in results.items()},indent=2))


if __name__=='__main__':
    run()
