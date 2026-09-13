"""Supplement the DP audit with independent baseline and episode accounting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.blend_budget_feedback_fixedq import BASE,OUT,DAYS,load,save,sha,source_checks
from experiments.exp008.verify import verify_arrays

ROOT=Path(__file__).resolve().parents[2]


def run():
    destination=OUT/'comparison_independent_audit.json'
    if destination.exists():
        raise FileExistsError('Completed supplement is immutable')
    source_checks()
    source=load(BASE/'dispatch.npz')
    table=pd.read_csv(OUT/'conditional_comparison.csv')
    actual=np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:,1:].to_numpy(float)
        for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')],axis=-1)
    records=[];maximum=0.
    for day in DAYS:
        p=load(OUT/f'day{day}_inputs.npz');i=day-31
        mode,power=int(p['initial_real_mode']),float(p['initial_power_kw'])
        prefix=6*(source['charge'][:i]-source['discharge'][:i]).ravel()
        nonidle=prefix[np.abs(prefix)>6e-6]
        assert mode==(int(np.sign(nonidle[-1])) if len(nonidle) else 1)
        for label in ('fixed_mask','DP'):
            flow=load(OUT/f'day{day}_{label}_actual_replay.npz')
            if label=='fixed_mask':
                for key,value in flow.items():
                    np.testing.assert_array_equal(value,source[key][i:i+1])
            detail=dict(original=p['q'][None,:],final=p['q'][None,:],actual=actual[day:day+1],
                        price=p['price'][None,:],**flow)
            checked=verify_arrays(detail,expected_days=1,start_day=day,initial_soc=float(p['initial_soc']),
                initial_mode=mode,initial_power_kw=power,source_actual=actual[day:day+1],source_price=p['price'][None,:])
            assert checked['passed'],checked
            row=table[(table.day==day)&(table.label==label)].iloc[0]
            sign=np.zeros(144,dtype=int)
            energy=flow['charge'][0]-flow['discharge'][0]
            sign[energy>1e-6]=1;sign[energy< -1e-6]=-1
            prior=1 if power>6e-6 else -1 if power< -6e-6 else 0
            charge=discharge=reverse=0;last_mode=mode
            for s in sign:
                charge+=int(s==1 and prior!=1)
                discharge+=int(s==-1 and prior!=-1)
                if s:
                    reverse+=int(s!=last_mode);last_mode=s
                prior=s
            expected={'charge_starts_with_previous_slot':charge,'discharge_starts_with_previous_slot':discharge,
                'episodes_with_previous_slot':charge+discharge,
                'direction_reversals_including_warmup_boundary':reverse,
                'active_slots':int(np.sum(sign!=0)),
                'throughput_kwh':float((flow['charge']+flow['discharge']).sum()),
                'final_soc':float(flow['states'][0,-1]),
                'total_cost':float(p['q']@p['price']+5*np.sum(flow['emergency'][0]*p['price']))}
            for key,value in expected.items():
                np.testing.assert_allclose(value,row[key],rtol=0,atol=1e-7)
            records.append({'day':day,'label':label,'metrics':expected,
                'source_arrays_or_DP_independent_physics_passed':True})
        historical=load(OUT/f'day{day}_fixed_mask_history.npz')
        for j,path in enumerate(p['all_net_paths']):
            soc=float(p['initial_soc'])
            for t in range(144):
                b=p['q'][t]-path[t]
                c=min(max(b,0),5000/6,max(0,(10800-soc)/np.sqrt(.9))) if p['allowed_charge'][t] else 0.
                d=min(max(-b,0),5000/6,max(0,(soc-1200)*np.sqrt(.9))) if not p['allowed_charge'][t] else 0.
                expected=(c,d,max(0,-b-d),max(0,b-c))
                for key,value in zip(('charge','discharge','emergency','surplus'),expected):
                    maximum=max(maximum,abs(value-historical[key][j,t]))
                soc+=np.sqrt(.9)*c-d/np.sqrt(.9)
                maximum=max(maximum,abs(soc-historical['states'][j,t+1]))
    assert maximum<1e-6
    source_checks()
    save(destination,{'passed':True,'records':records,'all140_fixed_mask_history_paths_scalar_replayed':True,
        'maximum_scalar_flow_error_kwh':maximum,'previous_mode_same_under_1e6_and6e6_power_thresholds':True,
        'previous_slot_aware_episode_and_nonidle_reversal_counts_independently_checked':True,
        'auditor_source_sha256':sha(Path(__file__)),'annual_goal_not_evaluated':True})
    (OUT/'comparison_auditor_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'comparison_audit_passed':True,'maximum_scalar_flow_error':maximum}))


if __name__=='__main__':
    run()
