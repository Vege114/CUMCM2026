"""Run the accepted Q3/Q4-3 LP with refund billing and own January startup.

Examples (from repository root):
  .venv/bin/python -m experiments.exp009.q34_run --scenario 3 --days 2 --label smoke
  .venv/bin/python -m experiments.exp009.q34_run --scenario 3
  .venv/bin/python -m experiments.exp009.q34_run --scenario 4-3
"""
import argparse
import copy
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import scipy

from experiments.exp008.planner import execute
from experiments.exp008.verify import battery_metrics
from experiments.exp009.q34_forecast import SelectedForecasts, residual_paths, DIRECTORY, MODEL
from experiments.exp009.q34_planner import plan, Settings, ETA, LOW, HIGH, LIMIT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp009'
SETTINGS = Settings(future_shortfall_weight=1.5)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settle(original, final, emergency, price):
    return np.stack((original*price, 1.5*price*np.maximum(final-original, 0),
        -0.5*price*np.maximum(original-final, 0), 5*price*emergency), axis=-1)


def verify(a, initial=6000.):
    """Slotwise physical and signed-refund accounting checks, no optimization."""
    c, d, e, w, q, g, s, p = (a[k] for k in
        ('charge', 'discharge', 'emergency', 'surplus', 'final', 'original', 'states', 'price'))
    balance = q+(a['actual'][..., 1]-a['actual'][..., 0])/6+d+e-c-w
    state_error = np.diff(s, axis=1)-ETA*c+d/ETA
    fees = np.stack((g*p, 1.5*p*np.maximum(q-g, 0), -0.5*p*np.maximum(g-q, 0), 5*p*e), axis=-1)
    max_balance, max_soc = float(np.abs(balance).max()), float(np.abs(state_error).max())
    continuity = float(np.abs(s[1:, 0]-s[:-1, -1]).max(initial=0))
    checks = {
        'all_finite': all(np.isfinite(a[k]).all() for k in a),
        'nonnegative_energy': min(float(a[k].min()) for k in ('charge','discharge','emergency','surplus','final','original')) >= -1e-6,
        'balance': max_balance < 1e-6, 'soc_recurrence': max_soc < 1e-6,
        'soc_continuity': continuity < 1e-6, 'initial_soc': abs(s[0, 0]-initial)<1e-6,
        'soc_bounds': bool(s.min()>=LOW-1e-6 and s.max()<=HIGH+1e-6),
        'power_bounds': bool(max(c.max(), d.max())<=LIMIT+1e-6),
        'no_simultaneous_actions': not bool(((c>1e-6)&(d>1e-6)).any()),
        'no_emergency_charging': not bool(((c>1e-6)&(e>1e-6)).any()),
        'signed_refund_billing': bool(np.max(np.abs(fees-a['fees']))<1e-6),
        'day_identifiers': bool(np.array_equal(a['days'], np.arange(int(a['days'][0]),int(a['days'][0])+len(s))))}
    if 'versions' in a:
        expected = np.concatenate([a['versions'][:, j, j*36:(j+1)*36] for j in range(4)],axis=1)
        checks['last_legal_version_per_executed_slot'] = bool(np.max(np.abs(expected-q))<1e-6)
        checks['original_is_midnight_version'] = bool(np.max(np.abs(a['versions'][:,0]-g))<1e-6)
    checks = {key: bool(value) for key, value in checks.items()}
    result = {'passed': all(checks.values()), 'checks': checks,
        'max_balance_error_kwh': max_balance, 'max_soc_error_kwh': max_soc,
        'max_cross_day_soc_error_kwh': continuity,
        'max_slot_fee_error_yuan': float(np.max(np.abs(fees-a['fees']))),
        'billing': dict(zip(('planned_cost','up_cost','down_refund','emergency_cost'), map(float,fees.sum(axis=(0,1))))),
        'total_cost': float(fees.sum()), 'planned_kwh': float(g.sum()),
        'final_regular_kwh': float(q.sum()), 'up_kwh': float(np.maximum(q-g,0).sum()),
        'down_kwh': float(np.maximum(g-q,0).sum()), 'emergency_kwh': float(e.sum()),
        'battery': battery_metrics(a, initial_mode=0, initial_power_kw=0.)}
    assert result['passed'], result
    return result


def causal_checks():
    rows=[]
    for scenario in ('3','4-3'):
        for day,slot in ((0,0),(0,36),(1,0),(1,36),(31,0),(32,36),(60,72),(180,108)):
            before=SelectedForecasts()
            changed=copy.copy(before.data)
            cutoff=day*144+slot
            changed.actual=before.data.actual.copy()
            changed.actual[cutoff:]+=[100000.,50000.,1000.]
            changed._forecasts={k:np.asarray(v).copy()+(90000. if k>cutoff else 0.) for k,v in before.data.forecasts.items()}
            after=SelectedForecasts(changed)
            a,b=before.get(day,slot,scenario),after.get(day,slot,scenario)
            for key in ('load_kw','pv_kw','price'):
                np.testing.assert_array_equal(a[key],b[key])
            ra,aa=residual_paths(before,day,slot,scenario)
            rb,ab=residual_paths(after,day,slot,scenario)
            np.testing.assert_array_equal(ra,rb)
            assert aa==ab and all(x<=day*144 for x in aa['label_stops_exclusive'])
            rows.append({'scenario':scenario,'day':day,'slot':slot,'future_mutation_identical':True,
                'same_issued_history_identical':True,'label_stop_max':max(aa['label_stops_exclusive'],default=None)})
    save(OUT/'q34_causality.json',{'passed':True,'rows':rows,
        'scope':'adapter/current official issue/price/cold-start/residual paths; HGB training provenance inherited unchanged'})


def summarize(a, directory, scenario, rows):
    whole=verify(a)
    save(directory/'verification_365.json',whole)
    formal_mask=a['days']>=31
    if formal_mask.any():
        formal={k:v[formal_mask] for k,v in a.items()}
        warm={k:v[~formal_mask] for k,v in a.items()}
        np.savez_compressed(directory/f'dispatch_{scenario}.npz',**formal)
        np.savez_compressed(directory/'warmup.npz',**warm)
        summary=verify(formal, float(formal['states'][0,0]))
        summary.update(scenario=scenario, days=len(formal['days']), intervals=len(formal['days'])*144,
            evaluation_start='2025-02-01',evaluation_end=str(pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(formal['days'][-1]))),
            initial_soc_january1_kwh=6000., initial_soc_february1_kwh=float(formal['states'][0,0]),
            january_cost=whole['total_cost']-summary['total_cost'],
            same_selected_controller_throughout_year=True, inherited_exp002_state=False,
            billing_rule='p*(g0+1.5*(q-g0)+-0.5*(g0-q)++5*e)',
            solver='7 representative historical paths shared-purchase LP, released every 6 hours',
            final_day_terminal_value=0., charge_deadband_kwh=20.)
        save(directory/'summary.json',summary)
        daily=pd.DataFrame(rows).query('day>=31')
        daily.to_csv(directory/'daily.csv',index=False)
        selected=daily[daily.date.isin(['2025-03-20','2025-06-21','2025-09-23','2025-12-21'])]
        selected.to_csv(directory/'specified_daily.csv',index=False)
        blocks=[]; emergencies=[]
        for day in (78,171,265,354):
            ids=np.flatnonzero(a['days']==day)
            if not len(ids): continue
            i=int(ids[0]); date=str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date())
            for start,stop in zip((0,24,48,72,96,120),(24,48,72,96,120,144)):
                blocks.append(dict(date=date,start_hour=start//6,end_hour=stop//6,
                    charge_kwh=float(a['charge'][i,start:stop].sum()),discharge_kwh=float(a['discharge'][i,start:stop].sum()),
                    soc_start_kwh=float(a['states'][i,start]),soc_end_kwh=float(a['states'][i,stop])))
            mask=a['emergency'][i]>1e-6
            edges=np.diff(np.r_[False,mask,False].astype(int));begins=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            for start,stop in zip(begins,ends):
                emergencies.append(dict(date=date,start_slot=int(start),stop_slot=int(stop),emergency_kwh=float(a['emergency'][i,start:stop].sum())))
        pd.DataFrame(blocks).to_csv(directory/'specified_battery.csv',index=False)
        pd.DataFrame(emergencies).to_csv(directory/'specified_emergency.csv',index=False)
        return summary
    return whole


def run(scenario, days=365, label=None):
    directory=OUT/('q3' if scenario=='3' else 'q4_3')
    if label: directory=directory/label
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'completion.json').exists():
        raise FileExistsError(f'Completed immutable run exists: {directory}')
    forecasts=SelectedForecasts()
    sources={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('q34_*.py')}
    sources.update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'experiments/exp008/planner.py',ROOT/'experiments/exp008/forecast.py',ROOT/'experiments/common/neural_v2/physics.py')})
    command=f'.venv/bin/python -m experiments.exp009.q34_run --scenario {scenario} --days {days}'+(f' --label {label}' if label else '')
    protocol={'scenario':scenario,'days':days,'initial_soc':6000.,'seed':42,'settings':asdict(SETTINGS),
        'charge_deadband_kwh':20.,'sources':sources,'raw_data_sha256':forecasts.data.hashes,
        'forecast_archive_sha256':digest(DIRECTORY/f'{MODEL}.npz'),'original_commit':'459cdb2',
        'changes':['downward half refund and q>=0','own selected algorithm January startup'],
        'command':command,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    save(directory/'protocol.json',protocol)
    began=perf_counter();soc=6000.;parts=[];rows=[];audits=[]
    for day in range(days):
        day_start=perf_counter();original=None;final=np.zeros(144);states=np.empty(145);states[0]=soc
        components={key:np.zeros(144) for key in ('charge','discharge','emergency','surplus')}
        versions=[]
        for slot,stop in zip((0,36,72,108),(36,72,108,144)):
            forecast=forecasts.get(day,slot,scenario)
            errors,risk=residual_paths(forecasts,day,slot,scenario)
            result=plan(forecast,float(states[slot]),errors,None if slot==0 else original[slot:],
                settings=SETTINGS,final_day=day==364,next_update_slot=stop-slot)
            if slot==0:original=result['purchase'].copy()
            final[slot:]=result['purchase'];versions.append(final.copy())
            actual=forecasts.data.actual[day*144+slot:day*144+stop]
            price=actual[:,2] if scenario=='4-3' else forecasts.data.fixed_price[slot:stop]
            executed=execute(final[slot:stop],actual,price,float(states[slot]),charge_deadband=20.)
            for key in components:components[key][slot:stop]=executed[key]
            states[slot:stop+1]=executed['states']
            audits.append({'day':day,'slot':slot,'forecast':forecast['audit'],'risk':risk,'solver':result['metadata']})
        actual=forecasts.data.actual[day*144:(day+1)*144].copy()
        price=actual[:,2].copy() if scenario=='4-3' else forecasts.data.fixed_price.copy()
        fees=settle(original,final,components['emergency'],price)
        parts.append(dict(original=original,final=final,**components,states=states,fees=fees,actual=actual,price=price,versions=np.stack(versions)))
        soc=float(states[-1]);billing=fees.sum(axis=0)
        rows.append(dict(day=day,date=str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
            scenario=scenario,planned_cost=float(billing[0]),up_cost=float(billing[1]),down_refund=float(billing[2]),
            emergency_cost=float(billing[3]),total_cost=float(fees.sum()),planned_kwh=float(original.sum()),
            final_regular_kwh=float(final.sum()),up_kwh=float(np.maximum(final-original,0).sum()),down_kwh=float(np.maximum(original-final,0).sum()),
            emergency_kwh=float(components['emergency'].sum()),initial_soc=float(states[0]),final_soc=soc,seconds=perf_counter()-day_start))
        if (day+1)%10==0 or day+1==days:
            print(json.dumps({'scenario':scenario,'days':day+1,'elapsed_seconds':perf_counter()-began,'cost':sum(r['total_cost'] for r in rows)}),flush=True)
            save(directory/'progress.json',{'days':day+1,'elapsed_seconds':perf_counter()-began,'last_day':rows[-1]})
    arrays={key:np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays['days']=np.arange(days)
    np.savez_compressed(directory/'dispatch_365.npz',**arrays)
    pd.DataFrame(rows).to_csv(directory/'daily_365.csv',index=False)
    save(directory/'audit.json',audits)
    summary=summarize(arrays,directory,scenario,rows)
    wall=perf_counter()-began
    save(directory/'completion.json',{'complete':True,'full_year':days==365,'wall_seconds':wall,
        'evaluation_cost':summary['total_cost'],'source_hashes':sources,'archive_sha256':digest(directory/'dispatch_365.npz'),
        'solver_planning_seconds':sum(a['solver']['planning_seconds'] for a in audits),
        'solver_feasible_all':all(a['solver']['feasible'] for a in audits),'run_command':command})
    print(json.dumps({'scenario':scenario,'done':True,'wall_seconds':wall,'summary':summary}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',choices=('3','4-3'),default='3')
    parser.add_argument('--days',type=int,default=365);parser.add_argument('--label');parser.add_argument('--causality',action='store_true')
    args=parser.parse_args()
    if args.causality:causal_checks()
    else:run(args.scenario,args.days,args.label)
