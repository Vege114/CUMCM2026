"""Fixed-five-day, fixed-Q adaptive DP diagnostic on the blend's own states."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.budget_feedback_diagnostic import check_flow, load, save, prefix_checks
from experiments.exp008.full28_mode_diagnostic_audit import sha
from experiments.exp008.planner import execute
from experiments.exp008.verify import battery_metrics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data/results/exp008/mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
OUT = ROOT/'data/results/exp008/budget_feedback/fixed5_blend_hold1_fixedQ_grid200_cap8'
DAYS = (31,90,151,243,364)


def archive_inputs():
    if OUT.exists():
        raise FileExistsError('Preserve existing fixed-Q evidence')
    OUT.mkdir(parents=True)
    protocol = {'days':DAYS,'model_id':'fixed_raw_HGB_ExtraTrees_half_Ridge28_memory',
        'base':str(BASE),'fixed_Q_from_each_blend_day':True,'Q_search_or_adjustment':False,
        'source_SOC_and_previous_REAL_nonidle_mode':'each selected day uses its own preceding blend execution prefix',
        'previous_planned_mode_not_used_as_previous_nonidle_mode':True,
        'state_grid_kwh':200.,'daily_actual_reversal_cap':8,'wear':.002,
        'terminal':'.45 except day364 zero',
        'history_days':28,'observed_error_bins':3,'conditional_emission_points':3,
        'all_five_models_and_DP_tables_locked_before_current_actual_read':True,
        'actual_suffix_mutation_stops':[1,36,108],
        'comparator':'original blend fixed-mask greedy execution and the same fixedQ all28 history',
        'grid_tuning_or_full_year_run':False,'development_not_independent_test':True,
        'annual_savings_not_evaluated':True,'final_model_selection':False}
    save(OUT/'protocol.json',protocol)
    inputs = [BASE/name for name in ('dispatch.npz','issued_forecasts.npz',
        'planning_audit.json','independent_audit.json','runtime_source_consistency_after.json')]
    inputs += [BASE/f'planning_day{d}.npz' for d in DAYS]
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        source = getattr(module,'__file__',None)
        if source:
            path = Path(source).resolve()
            if path.is_relative_to(ROOT/'experiments') and path.suffix == '.py':
                paths.add(path)
    sources = {}
    for path in sorted(paths):
        name = str(path.relative_to(ROOT))
        target = OUT/'source_archive'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        sources[name] = sha(path)
    save(OUT/'source_manifest.json',{'protocol_sha256':sha(OUT/'protocol.json'),
        'sources':sources,'inputs':{str(p):sha(p) for p in inputs}})
    save(OUT/'functional_checks.json',policy.checks())


def source_checks():
    manifest = json.loads((OUT/'source_manifest.json').read_text())
    assert sha(OUT/'protocol.json') == manifest['protocol_sha256']
    for path,digest in manifest['inputs'].items():
        assert sha(Path(path)) == digest
    for path,digest in manifest['sources'].items():
        assert sha(ROOT/path) == sha(OUT/'source_archive'/path) == digest


def fixed_mask_history(q,p):
    parts = []
    for path in p['all_net_paths']:
        channels = 6*np.column_stack((np.maximum(path,0),np.maximum(-path,0)))
        flow = execute(q,channels,p['price'],float(p['initial_soc']),
                       charge_mask=p['allowed_charge'],charge_deadband=0.)
        flow['fees'] = np.stack((q*p['price'],np.zeros(144),np.zeros(144),
                                5*flow['emergency']*p['price']),axis=-1)
        parts.append(flow)
    return {key:np.stack([p[key] for p in parts]) for key in parts[0]}


def metric_row(day,label,q,p,flow):
    metrics = battery_metrics(flow,initial_mode=int(p['initial_real_mode']),
                              initial_power_kw=float(p['initial_power_kw']))
    signs = np.where(np.abs(flow['charge'][0]-flow['discharge'][0])>1e-6,
                     np.sign(flow['charge'][0]-flow['discharge'][0]),0).astype(int)
    previous = int(np.sign(p['initial_power_kw'])) if abs(p['initial_power_kw'])>6e-6 else 0
    prior = np.r_[previous,signs[:-1]]
    starts_c = int(np.sum((signs==1)&(prior!=1)))
    starts_d = int(np.sum((signs==-1)&(prior!=-1)))
    return {'day':day,'label':label,'total_cost':float(flow['fees'].sum()),
        'planned_cost':float(q@p['price']),'emergency_cost':float(flow['fees'][:,:,3].sum()),
        'charge_starts_with_previous_slot':starts_c,'discharge_starts_with_previous_slot':starts_d,
        'episodes_with_previous_slot':starts_c+starts_d,**metrics}


def run():
    from experiments.exp008 import blend_budget_feedback_fixedq_audit  # freeze auditor before run
    archive_inputs()
    assert json.loads((BASE/'independent_audit.json').read_text())['passed']
    prepared = {}
    # Only selected causal policy/state fields are read from the old archive;
    # the actual array and realized target-day fees/actions are read after lock.
    with np.load(BASE/'dispatch.npz') as source, np.load(BASE/'issued_forecasts.npz') as issued:
        for day in DAYS:
            i = day-31
            p = load(BASE/f'planning_day{day}.npz')
            q = source['original'][i].copy()
            prefix = 6*(source['charge'][:i]-source['discharge'][:i]).ravel()
            nonidle = prefix[np.abs(prefix)>1e-6]
            p['initial_real_mode'] = int(np.sign(nonidle[-1])) if len(nonidle) else 1
            p['initial_power_kw'] = float(prefix[-1]) if len(prefix) else 172.75999999999976
            assert float(p['initial_soc']) == source['states'][i,0]
            forecast = (issued['values'][i,:,0]-issued['values'][i,:,1])/6
            errors = p['all_net_paths']-forecast[None,:]
            origins = np.arange(day-28,day)*144
            model = policy.fit_error_model(forecast,errors,points=3,history_origins=origins,cutoff=day*144)
            terminal = 0. if day==364 else .45
            grid,values,metadata = policy.value_functions(q,model,p['price'],
                grid_kwh=200.,wear=.002,terminal=terminal,budget=8)
            history = policy.execute_paths(q,p['all_net_paths'],p['price'],float(p['initial_soc']),
                grid,values,model,wear=.002,initial_mode=int(p['initial_real_mode']),budget=8)
            check_flow(q,p['all_net_paths'],p['price'],history,float(p['initial_soc']),int(p['initial_real_mode']))
            fixed = fixed_mask_history(q,p)
            check_flow(q,p['all_net_paths'],p['price'],fixed,float(p['initial_soc']),int(p['initial_real_mode']))
            mutations = prefix_checks(q,p,grid,values,model,history)
            np.savez_compressed(OUT/f'day{day}_inputs.npz',**p,q=q,forecast_net=forecast,
                                historical_errors=errors,history_origins=origins)
            np.savez_compressed(OUT/f'day{day}_model.npz',**{k:v for k,v in model.items() if k!='metadata'})
            np.savez_compressed(OUT/f'day{day}_policy.npz',q=q,grid=grid,values=values)
            np.savez_compressed(OUT/f'day{day}_DP_history.npz',**history)
            np.savez_compressed(OUT/f'day{day}_fixed_mask_history.npz',**fixed)
            history_score = {label:policy.historical_objective(flow,wear=.002,terminal=terminal)
                             for label,flow in [('DP',history),('fixed_mask',fixed)]}
            save(OUT/f'day{day}_history.json',{'model':model['metadata'],'DP':metadata,
                'full28_historical_objective':history_score,
                'full28_historical_cash_fee':{label:float(flow['fees'].sum(axis=(1,2)).mean())
                    for label,flow in [('DP',history),('fixed_mask',fixed)]},
                'DP_initial_expected_objective':float(q@p['price']+policy.initial_value(grid,values,model,
                    float(p['initial_soc']),mode=int(p['initial_real_mode']),remaining=8)),
                'historical_prefix_mutations':mutations})
            prepared[day] = (p,q,model,grid,values)
            print('LOCKED_FIXED_Q_POLICY',day,history_score,flush=True)
    locked = {str(p.relative_to(OUT)):sha(p) for p in OUT.glob('day*') if p.is_file()}
    save(OUT/'all_policies_locked_before_actual.json',{'actual_array_or_CSV_loaded':False,
        'fixed_Q_no_selection':True,'files':locked})
    actual = np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:,1:].to_numpy(float)
        for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')],axis=-1)
    rows = []
    with np.load(BASE/'dispatch.npz') as source:
        for day,(p,q,model,grid,values) in prepared.items():
            net = ((actual[day,:,0]-actual[day,:,1])/6)[None,:]
            fixed = {key:source[key][day-31:day-30].copy() for key in
                     ('charge','discharge','emergency','surplus','states','fees')}
            dp = policy.execute_paths(q,net,p['price'],float(p['initial_soc']),grid,values,model,
                                      wear=.002,initial_mode=int(p['initial_real_mode']),budget=8)
            for label,flow in [('fixed_mask',fixed),('DP',dp)]:
                check_flow(q,net,p['price'],flow,float(p['initial_soc']),int(p['initial_real_mode']))
                np.savez_compressed(OUT/f'day{day}_{label}_actual_replay.npz',**flow)
                rows.append(metric_row(day,label,q,p,flow))
    for name,digest in locked.items():
        assert sha(OUT/name) == digest
    source_checks()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'conditional_comparison.csv',index=False)
    keys = ['total_cost','planned_cost','emergency_cost','throughput_kwh','active_slots',
            'direction_reversals_including_warmup_boundary','episodes_with_previous_slot',
            'charge_starts_with_previous_slot','discharge_starts_with_previous_slot',
            'final_soc','power_ramp_total_kw_including_warmup_boundary']
    delta = frame[frame.label=='DP'].set_index('day')[keys]-frame[frame.label=='fixed_mask'].set_index('day')[keys]
    delta.to_csv(OUT/'conditional_deltas.csv')
    save(OUT/'summary.json',{'complete':True,'days':DAYS,'fixed_Q':True,
        'paired_deltas':delta.reset_index().to_dict('records'),
        'totals_without_terminal_soc':{label:frame[frame.label==label][[k for k in keys if k!='final_soc']].sum().to_dict()
            for label in ('fixed_mask','DP')},
        'all_models_and_DP_tables_locked_before_actual':True,
        'all_input_and_source_files_unchanged':True,'annual_savings_not_evaluated':True,
        'not_a_continuous5day_evaluation':True,'own_initial_states_not_propagated_between_selected_days':True,
        'development_not_independent_test':True,'goal_not_evaluated':True})
    blend_budget_feedback_fixedq_audit.run()


if __name__=='__main__':
    run()
