"""Fixed-five-day control test after the local-clock probability score gate."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.budget_feedback_diagnostic import check_flow, load
from experiments.exp008.budget_feedback_diagnostic_audit import scalar_bellman
from experiments.exp008.budget_grid_diagnostic import numerical_metrics
from experiments.exp008.local_transition_prior import ROOT, local_clock_model, sha
from experiments.problem2.exp003.data import Data

BASE = ROOT/'data/results/exp008/budget_feedback/fixed5_hgb_cap8_grid200_blocks8_step50'
OUT = ROOT/'data/results/exp008/local_transition_prior/fixed5_control'
DAYS = (31, 90, 151, 243, 364)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def metrics(q, p, flow, day):
    result = numerical_metrics(q, p, flow, day)
    power = flow['charge']-flow['discharge']
    activity = np.where(np.abs(power)>1e-6, np.sign(power), 0)
    boundary = float(p['initial_power_kw']) / 6
    previous = np.concatenate((np.full((len(power),1), np.sign(boundary) if abs(boundary)>1e-6 else 0), activity[:,:-1]), axis=1)
    result['activity_episode_starts'] = int(np.sum((activity != 0) & (activity != previous)))
    return result


def run():
    scores = json.loads((OUT.parent/'summary.json').read_text())
    assert scores['complete'] and scores['both_scores_improve']
    OUT.mkdir(exist_ok=False)
    sources = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and p.suffix == '.py' and '.venv' not in p.parts:
                rel = p.relative_to(ROOT)
                target = OUT/'sources'/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target); sources[str(rel)] = sha(p)
    required = [BASE/f'day{day}_{suffix}.npz' for day in DAYS for suffix in ('inputs','model','initial_policy','initial_actual_replay')]
    inputs = {str(p): sha(p) for p in required}
    save(OUT/'protocol.json', {'days': DAYS, 'candidate_count': 1, 'change': 'clock-local transition shrinkage prior only',
        'Q_SOC_previous_real_mode_fixed_to_original_five_day_inputs': True, 'grid_kwh': 200, 'budget': 8,
        'wear': .002, 'terminal': '.45 except final calendar day 0', 'no_annual_extrapolation': True,
        'not_an_untouched_test': True, 'score_gate': scores, 'source_sha256': sources, 'input_sha256': inputs})
    prepared, history, bellman = {}, [], []
    for day in DAYS:
        p = load(BASE/f'day{day}_inputs.npz'); original = load(BASE/f'day{day}_initial_policy.npz')
        old, new = local_clock_model(p['forecast_net'], p['historical_errors'], p['history_origins'], day*144)
        signed_model = load(BASE/f'day{day}_model.npz')
        for key, value in signed_model.items():
            np.testing.assert_array_equal(value, old[key])
        q = original['q']; terminal = 0. if day == 364 else .45
        grid, values, _ = policy.value_functions(q, new, p['price'], grid_kwh=200, terminal=terminal)
        new_history = policy.execute_paths(q, p['all_net_paths'], p['price'], float(p['initial_soc']), grid, values, new, initial_mode=int(p['initial_real_mode']))
        old_history = policy.execute_paths(q, p['all_net_paths'], p['price'], float(p['initial_soc']), original['grid'], original['values'], old, initial_mode=int(p['initial_real_mode']))
        check_flow(q, p['all_net_paths'], p['price'], new_history, float(p['initial_soc']), int(p['initial_real_mode']))
        for t in (0,35,71,107,143):
            for eb in range(3):
                for mi in range(2):
                    for remaining in (0,4,8):
                        for si in (0,len(grid)//2,len(grid)-1):
                            v = scalar_bellman(q,p['price'],grid,values,new,t,eb,mi,remaining,si)
                            bellman.append(abs(v-values[t,eb,mi,remaining,si]))
        history.append({'day': day, 'old': policy.historical_objective(old_history,terminal=terminal), 'new': policy.historical_objective(new_history,terminal=terminal)})
        np.savez_compressed(OUT/f'day{day}_policy.npz',q=q,grid=grid,values=values)
        np.savez_compressed(OUT/f'day{day}_model.npz',**{k:v for k,v in new.items() if k!='metadata'})
        np.savez_compressed(OUT/f'day{day}_history.npz',**new_history)
        prepared[day] = p,original,old,new,q,grid,values
    assert max(bellman)<1e-7
    save(OUT/'all_policies_locked.json', {'actual_data_loaded': False, 'policy_sha256': {str(day):sha(OUT/f'day{day}_policy.npz') for day in DAYS},
        'model_sha256': {str(day):sha(OUT/f'day{day}_model.npz') for day in DAYS}, 'scalar_bellman_count':len(bellman),'scalar_bellman_max_error':max(bellman)})
    data = Data(); rows=[]
    for day,(p,original,old,new,q,grid,values) in prepared.items():
        raw=data.actual[day*144:(day+1)*144,:2]; net=((raw[:,0]-raw[:,1])/6)[None,:]
        baseline=policy.execute_paths(q,net,p['price'],float(p['initial_soc']),original['grid'],original['values'],old,initial_mode=int(p['initial_real_mode']))
        candidate=policy.execute_paths(q,net,p['price'],float(p['initial_soc']),grid,values,new,initial_mode=int(p['initial_real_mode']))
        signed=load(BASE/f'day{day}_initial_actual_replay.npz')
        for key,value in signed.items(): np.testing.assert_array_equal(baseline[key],value)
        for label,flow in [('original',baseline),('local',candidate)]:
            check_flow(q,net,p['price'],flow,float(p['initial_soc']),int(p['initial_real_mode']))
            np.testing.assert_allclose(flow['fees'].sum(), q@p['price']+5*(flow['emergency']*p['price']).sum(),atol=1e-7,rtol=0)
            rows.append({'model':label,**metrics(q,p,flow,day)})
        np.savez_compressed(OUT/f'day{day}_actual.npz',**candidate)
        for stop in (1,36,108):
            changed=net.copy();changed[:,stop:]+=np.linspace(-2e4,5e4,144-stop)
            mutated=policy.execute_paths(q,changed,p['price'],float(p['initial_soc']),grid,values,new,initial_mode=int(p['initial_real_mode']))
            for key in ('charge','discharge','emergency','surplus','observed_error_bins'): np.testing.assert_array_equal(mutated[key][:,:stop],candidate[key][:,:stop])
            for key in ('states','modes','remaining_budget'): np.testing.assert_array_equal(mutated[key][:,:stop+1],candidate[key][:,:stop+1])
    assert all(sha(ROOT/p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in inputs.items())
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'metrics.csv',index=False)
    delta=frame[frame.model=='local'].set_index('day').drop(columns='model')-frame[frame.model=='original'].set_index('day').drop(columns='model')
    delta.to_csv(OUT/'deltas.csv')
    result={'complete':True,'days':DAYS,'fixed5_fee_change':float(delta.cash_fee.sum()),'historical_objective_change':sum(r['new']-r['old'] for r in history),
        'direction_reversals_change':int(delta.direction_reversals_including_boundary.sum()),'episodes_change':int(delta.activity_episode_starts.sum()),
        'throughput_change_kwh':float(delta.throughput_kwh.sum()),'all_sources_unchanged':True,'future_prefix_checks':15,
        'baseline_signed_flows_exact':True,'scalar_bellman_checks':len(bellman),'max_scalar_bellman_error':max(bellman),
        'annual_result':False,'final_selection':False}
    save(OUT/'historical_objectives.json',history);save(OUT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__': run()
