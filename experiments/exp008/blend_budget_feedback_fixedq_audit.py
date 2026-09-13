"""Independent fixed-Q identity, scalar Bellman, fee and prefix checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.budget_feedback_diagnostic_audit import scalar_bellman
from experiments.exp008.blend_budget_feedback_fixedq import BASE, OUT, DAYS, load, save, sha, source_checks
from experiments.exp008.verify import verify_arrays

ROOT = Path(__file__).resolve().parents[2]


def run():
    protected = {str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    source_checks()
    locked = json.loads((OUT/'all_policies_locked_before_actual.json').read_text())
    assert not locked['actual_array_or_CSV_loaded'] and locked['fixed_Q_no_selection']
    for name,digest in locked['files'].items():
        assert sha(OUT/name) == digest
    actual = np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:,1:].to_numpy(float)
        for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')],axis=-1)
    source = load(BASE/'dispatch.npz')
    issued = load(BASE/'issued_forecasts.npz')
    records, maximum, bellman_count = [], 0., 0
    for day in DAYS:
        i = day-31
        p = load(OUT/f'day{day}_inputs.npz')
        before = load(BASE/f'planning_day{day}.npz')
        np.testing.assert_array_equal(p['q'],source['original'][i])
        assert float(p['initial_soc']) == source['states'][i,0]
        prefix = 6*(source['charge'][:i]-source['discharge'][:i]).ravel()
        active = prefix[np.abs(prefix)>1e-6]
        mode = int(np.sign(active[-1])) if len(active) else 1
        power = float(prefix[-1]) if len(prefix) else 172.75999999999976
        assert mode==p['initial_real_mode'] and power==p['initial_power_kw']
        np.testing.assert_array_equal(before['all_net_paths'],p['all_net_paths'])
        forecast = (issued['values'][i,:,0]-issued['values'][i,:,1])/6
        np.testing.assert_array_equal(forecast,p['forecast_net'])
        np.testing.assert_array_equal(p['historical_errors'],p['all_net_paths']-forecast[None,:])
        np.testing.assert_array_equal(p['history_origins'],np.arange(day-28,day)*144)
        model = policy.fit_error_model(forecast,p['historical_errors'],points=3,
            history_origins=p['history_origins'],cutoff=day*144)
        frozen_model = load(OUT/f'day{day}_model.npz')
        for key,value in model.items():
            if key!='metadata':
                np.testing.assert_array_equal(value,frozen_model[key])
        terminal = 0. if day==364 else .45
        grid,values,_ = policy.value_functions(p['q'],model,p['price'],
            grid_kwh=200.,wear=.002,terminal=terminal,budget=8)
        frozen = load(OUT/f'day{day}_policy.npz')
        for key,value in [('q',p['q']),('grid',grid),('values',values)]:
            np.testing.assert_array_equal(value,frozen[key])
        scores = json.loads((OUT/f'day{day}_history.json').read_text())
        for label in ('DP','fixed_mask'):
            flow = load(OUT/f'day{day}_{label}_history.npz')
            exact = p['q']@p['price']+np.mean(np.sum(5*flow['emergency']*p['price']
                +.002*(flow['charge']+flow['discharge']),axis=1)-terminal*(flow['states'][:,-1]-1200.))
            np.testing.assert_allclose(exact,scores['full28_historical_objective'][label],rtol=0,atol=1e-6)
            if label=='DP':
                replay = policy.execute_paths(p['q'],p['all_net_paths'],p['price'],float(p['initial_soc']),
                    grid,values,model,wear=.002,initial_mode=mode,budget=8)
                for key,value in replay.items():
                    np.testing.assert_array_equal(value,flow[key])
        for t in (0,36,72,108,143):
            for previous_bin in range(3):
                for mode_index in range(2):
                    for remaining in (0,4,8):
                        for soc_index in (0,24,48):
                            v = scalar_bellman(p['q'],p['price'],grid,values,model,t,
                                              previous_bin,mode_index,remaining,soc_index)
                            error = abs(v-values[t,previous_bin,mode_index,remaining,soc_index])
                            maximum = max(maximum,float(error));bellman_count += 1
                            assert error < 1e-6
        net = ((actual[day,:,0]-actual[day,:,1])/6)[None,:]
        flow = policy.execute_paths(p['q'],net,p['price'],float(p['initial_soc']),grid,values,model,
            wear=.002,initial_mode=mode,budget=8)
        frozen = load(OUT/f'day{day}_DP_actual_replay.npz')
        for key,value in flow.items():
            np.testing.assert_array_equal(value,frozen[key])
        detail = dict(original=p['q'][None,:],final=p['q'][None,:],actual=actual[day:day+1],
                      price=p['price'][None,:],**flow)
        checked = verify_arrays(detail,expected_days=1,start_day=day,initial_soc=float(p['initial_soc']),
            initial_mode=mode,initial_power_kw=power,source_actual=actual[day:day+1],source_price=p['price'][None,:])
        assert checked['passed'],checked
        assert checked['battery_metrics']['direction_reversals_including_warmup_boundary'] <= 8
        for stop in (1,36,108):
            changed = net.copy();changed[:,stop:] += np.linspace(1e4,7e4,144-stop)
            replay = policy.execute_paths(p['q'],changed,p['price'],float(p['initial_soc']),grid,values,model,
                wear=.002,initial_mode=mode,budget=8)
            for key in ('charge','discharge','emergency','surplus','observed_error_bins'):
                np.testing.assert_array_equal(flow[key][:,:stop],replay[key][:,:stop])
            for key in ('states','modes','remaining_budget'):
                np.testing.assert_array_equal(flow[key][:,:stop+1],replay[key][:,:stop+1])
        records.append({'day':day,'fixed_Q_and_own_blend_initial_state_verified':True,
            'all28_history_and_DP_table_reconstructed_exactly':True,'independent_actual_verification':checked,
            'future_actual_prefix_mutations_passed':3})
    source_checks()
    assert protected == {str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'independent_audit.json',{'passed':True,'days':DAYS,'records':records,
        'scalar_Bellman_checks':bellman_count,'maximum_scalar_Bellman_error':maximum,
        'all_source_inputs_and_produced_outputs_preserved':True,
        'raw_model_causality_inherited_from_signed_blend_full334_audit':True,
        'annual_goal_not_evaluated':True,'development_not_independent_test':True})
    print(json.dumps({'independent_audit_passed':True,'scalar_Bellman_checks':bellman_count,
                      'maximum_error':maximum},indent=2),flush=True)


if __name__=='__main__':
    run()
