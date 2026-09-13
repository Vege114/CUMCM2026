"""Single predeclared exp008 -> exp009 Q2 objective treatment on frozen forecasts."""
from __future__ import annotations
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from experiments.exp008.closed_loop import optimize
from experiments.exp008.planner import execute
from experiments.exp008.verify import verify_npz
from experiments.problem2.exp003.data import ROOT, Data
from experiments.exp009.q2_cost_model import plan

BASE=ROOT/'data/results/exp008/mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
OUT=ROOT/'data/results/exp009/q2_cost_only'
INITIAL=1421.7991105135516


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def checkpoint(parts,audits,rows):
    arrays={key:np.stack([p[key] for p in parts]) for key in parts[0]}
    arrays['days']=np.arange(31,31+len(parts))
    np.savez_compressed(OUT/'dispatch.npz',**arrays)
    save(OUT/'planning_audit.json',audits)
    pd.DataFrame(rows).to_csv(OUT/'daily.csv',index=False)
    return arrays


def run():
    if OUT.exists():raise FileExistsError('Never overwrite or resume a different trial')
    OUT.mkdir(parents=True)
    data=Data();global_protocol=ROOT/'experiments/exp009/protocol.json'
    source_paths={Path(__file__).resolve(),global_protocol}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(ROOT/'experiments') and p.suffix=='.py':source_paths.add(p)
    frozen={str(p.relative_to(ROOT)):digest(p) for p in sorted(source_paths)}
    for path in source_paths:
        target=OUT/'source_archive'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    input_paths=[BASE/'dispatch.npz',BASE/'planning_audit.json',BASE/'issued_forecasts.npz']+[BASE/f'planning_day{d}.npz' for d in range(31,365)]
    inputs={str(p.relative_to(ROOT)):digest(p) for p in input_paths}
    save(OUT/'provenance.json',{'protocol_sha256':digest(global_protocol),'source_sha256':frozen,'input_sha256':inputs,
        'data_hashes':data.hashes,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'platform':platform.platform(),'forecast_reused':True,'training_seconds':0,'forecast_reconstruction_seconds':0,
        'old_endogenous_decisions_not_used':['purchase','actual_or_scenario_states','charge','discharge','old_mask'],
        'predeclared_treatment':'remove only cap8 constraint and wear=.002 coefficients; identical model method/physical constraints/time budget/refinement'})
    old_audits=json.loads((BASE/'planning_audit.json').read_text())
    soc,mode,age=INITIAL,1,1;parts=[];audits=[];rows=[];began=perf_counter()
    for day in range(31,365):
        tick=perf_counter()
        with np.load(BASE/f'planning_day{day}.npz') as z:
            paths=z['all_net_paths'].copy();price=z['price'].copy();indices=z['selected_scenario_indices'].copy()
        if not np.array_equal(price,data.fixed_price):raise AssertionError('Changed fixed tariff')
        if not np.array_equal(indices,np.linspace(0,len(paths)-1,3).astype(int)):raise AssertionError('Changed scenario selection')
        try:initial=plan(paths[indices],price,soc,mode,age,wear=0.,final=day==364)
        except (RuntimeError,AssertionError) as error:
            if parts:checkpoint(parts,audits,rows)
            save(OUT/'failure.json',{'day':day,'completed_days':len(parts),'error':str(error),'parameter_retuning':False});raise
        meta=initial['metadata']
        np.savez_compressed(OUT/f'planning_day{day}.npz',**{k:v for k,v in initial.items() if k!='metadata'},
            initial_soc=soc,previous_planned_mode=mode,previous_planned_run_slots=age,
            source_day=day,selected_scenario_indices=indices)
        refined=optimize(initial['purchase'],paths,price,soc,charge_mask=initial['allowed_charge'],throughput=0.,variation=0.,
            terminal=0. if day==364 else .45,deadband=0.,maxiter=120)
        q=refined['purchase'];actual=data.actual[day*144:(day+1)*144]
        detail=execute(q,actual,price,soc,charge_mask=initial['allowed_charge'],charge_deadband=0.)
        detail.update(original=q.copy(),final=q.copy(),actual=actual.copy(),price=price.copy(),allowed_charge=initial['allowed_charge'].copy())
        detail['fees']=np.stack((q*price,np.zeros(144),np.zeros(144),5*detail['emergency']*price),axis=-1)
        parts.append(detail)
        old=old_audits[day-31]
        audits.append({'day':day,'forecast':old['forecast'],'history':old['history'],'mip':meta,'greedy_refinement':refined['metadata'],
            'external_source_path':str((BASE/f'planning_day{day}.npz').relative_to(ROOT)),
            'external_source_sha256':inputs[str((BASE/f'planning_day{day}.npz').relative_to(ROOT))],
            'planned_boundary_state_before':[mode,age],'planned_boundary_state_after':[meta['final_planned_mode'],meta['final_planned_run_slots']],
            'same_issued_predictions_and_historical_paths_as_exp008':True})
        rows.append({'day':day,'date':str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
            'total_cost':float(detail['fees'].sum()),'planned_cost':float(detail['fees'][:,0].sum()),'emergency_cost':float(detail['fees'][:,3].sum()),
            'initial_soc':soc,'final_soc':float(detail['states'][-1]),'mip_gap':meta['mip_gap'],'mip_seconds':meta['seconds'],
            'refinement_seconds':refined['metadata']['planning_seconds'],'planned_changes':meta['planned_changes_including_boundary'],
            'seconds':perf_counter()-tick})
        soc,mode,age=rows[-1]['final_soc'],meta['final_planned_mode'],meta['final_planned_run_slots']
        if len(parts)==3 or len(parts)%10==0 or day==364:
            checkpoint(parts,audits,rows)
            print(json.dumps({'completed_days':len(parts),'total_cost_so_far':sum(r['total_cost'] for r in rows),'last_gap':meta['mip_gap']}),flush=True)
        if len(parts)==3:
            gate=verify_npz(OUT/'dispatch.npz',expected_days=3,audit_path=OUT/'planning_audit.json')
            save(OUT/'first3_physics.json',gate)
            if not gate['passed']:raise AssertionError(gate['errors'])
    wall=perf_counter()-began
    verification=verify_npz(OUT/'dispatch.npz',expected_days=334,audit_path=OUT/'planning_audit.json')
    if not verification['passed']:raise AssertionError(verification['errors'])
    for path,expected in {**frozen,**inputs}.items():
        if digest(ROOT/path)!=expected:raise AssertionError('Source changed during fixed run: '+path)
    save(OUT/'summary.json',{'complete':True,'days':334,'verification':verification,'total_cost':verification['recomputed_total_cost'],
        'battery':verification['battery_metrics'],'wall_seconds':wall,'forecast_unchanged_days':334,
        'mip_seconds':sum(r['mip_seconds'] for r in rows),'refinement_seconds':sum(r['refinement_seconds'] for r in rows),
        'cost_objective_only':True,'count_is_diagnostic_only':True,'all_sources_unchanged':True})
    print(json.dumps({'complete':True,'cost':verification['recomputed_total_cost'],'battery':verification['battery_metrics'],'wall_seconds':wall}),flush=True)


if __name__=='__main__':run()
