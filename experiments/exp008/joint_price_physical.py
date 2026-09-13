"""Paired-price variant of the frozen physical shared-mode initializer.

Feasibility constraints are copied unchanged from mode_planning_physical.plan;
only shared-purchase and scenario-emergency prices change. Scenario terminal
value still uses the current forecast price. Recourse remains optimistic.
"""

import numpy as np

from experiments.common.neural_v2.physics import LinearModel
from experiments.exp008.planner import ETA, HIGH, LIMIT, LOW


def plan(net_paths, price, soc, previous_mode=1, *, block_slots=6,
         switching=50., wear=.002, seconds=5., gap=.002, final=False,
         scenario_prices=None):
    paths=np.asarray(net_paths,float)
    price=np.asarray(price,float)
    k,n=paths.shape
    if price.shape!=(n,) or not np.all(price>0) or not np.isfinite(paths).all():
        raise ValueError('finite scenario net kWh and matching positive price required')
    if not LOW<=soc<=HIGH or block_slots<1:
        raise ValueError('invalid initial SOC or block length')
    paired_prices = (np.broadcast_to(price, (k,n)).copy() if scenario_prices is None
                     else np.asarray(scenario_prices, float))
    if paired_prices.shape != (k,n) or not np.isfinite(paired_prices).all() or not np.all(paired_prices > 0):
        raise ValueError('scenario_prices must be a finite positive matrix matching net_paths')
    mean_price = (paired_prices[0].copy() if np.all(paired_prices == paired_prices[0])
                  else paired_prices.mean(axis=0))
    nb=(n+block_slots-1)//block_slots
    m=LinearModel()
    # Larger common purchases are entirely wasted in every scenario and are
    # dominated under strictly positive fixed-plan purchase prices.
    q_upper=np.maximum(paths.max(axis=0),0.)+LIMIT
    q=m.variables((n,),upper=q_upper,cost=mean_price)
    mode=m.variables((nb,),upper=1.,integer=True)
    flip=m.variables((nb,),cost=switching)
    c=m.variables((k,n),upper=LIMIT,cost=wear/k)
    d=m.variables((k,n),upper=LIMIT,cost=wear/k)
    # With emergency energy unable to charge, q>=0 and discharge>=0,
    # physically useful emergency energy cannot exceed positive net demand.
    emergency_upper=np.maximum(paths,0.)
    e=m.variables((k,n),upper=emergency_upper,cost=5*paired_prices/k)
    w=m.variables((k,n))
    s=m.variables((k,n),lower=LOW,upper=HIGH)
    may_charge=m.variables((k,n),upper=1.,integer=True)
    for idx in s[:,-1]:
        m.objective[int(idx)]=0. if final else -float(price.min())/ETA/k
    for b in range(nb):
        terms=[(mode[b],1.)]
        rhs=float(previous_mode>0) if b==0 else 0.
        if b:
            terms.append((mode[b-1],-1.))
        m.constraint(terms+[(flip[b],-1.)],upper=rhs)
        m.constraint([(i,-v) for i,v in terms]+[(flip[b],-1.)],upper=-rhs)
    for j in range(k):
        for t in range(n):
            m.constraint([(q[t],1),(c[j,t],-1),(d[j,t],1),(e[j,t],1),(w[j,t],-1)],
                         paths[j,t],paths[j,t])
            terms=[(s[j,t],1),(c[j,t],-ETA),(d[j,t],1/ETA)]
            if t:
                terms.append((s[j,t-1],-1.))
            rhs=soc if t==0 else 0.
            m.constraint(terms,rhs,rhs)
            block=t//block_slots
            m.constraint([(c[j,t],1),(mode[block],-LIMIT)],upper=0.)
            m.constraint([(d[j,t],1),(mode[block],LIMIT)],upper=LIMIT)
            m.constraint([(c[j,t],1),(may_charge[j,t],-LIMIT)],upper=0.)
            m.constraint([(e[j,t],1),(may_charge[j,t],emergency_upper[j,t])],
                         upper=emergency_upper[j,t])
    x,metadata=m.solve(seconds=seconds,gap=gap)
    if x is None:
        raise RuntimeError(f'No feasible physical shared-mode incumbent: {metadata}')
    states=np.column_stack((np.full(k,soc),x[s]))
    balance=x[q][None,:]+x[d]+x[e]-x[c]-x[w]-paths
    state_error=np.diff(states,axis=1)-ETA*x[c]+x[d]/ETA
    check={'maximum_balance_error_kwh':float(np.abs(balance).max()),
           'maximum_soc_equation_error_kwh':float(np.abs(state_error).max()),
           'emergency_charging_slots':int(np.sum((x[c]>1e-6)&(x[e]>1e-6))),
           'simultaneous_charge_discharge_slots':int(np.sum((x[c]>1e-6)&(x[d]>1e-6))),
           'minimum_soc_kwh':float(states.min()),'maximum_soc_kwh':float(states.max()),
           'maximum_charge_or_discharge_kwh':float(max(x[c].max(),x[d].max())),
           'minimum_variable':float(x.min())}
    check['passed']=(max(check['maximum_balance_error_kwh'],check['maximum_soc_equation_error_kwh'])<1e-6
                     and check['emergency_charging_slots']==0
                     and check['simultaneous_charge_discharge_slots']==0
                     and states.min()>=LOW-1e-6 and states.max()<=HIGH+1e-6
                     and check['maximum_charge_or_discharge_kwh']<=LIMIT+1e-6
                     and check['minimum_variable']>=-1e-6)
    if not check['passed']:
        raise AssertionError(check)
    metadata.update(method='paired_price_shared_mode_milp_without_emergency_charging',
                    paired_price_and_net_paths=True,
                    common_purchase_price='mean_scenario_price',
                    terminal_price='minimum_current_forecast_price_unchanged',
                    scenario_count=k,nonanticipative_recourse_certificate=False,
                    planned_mode_changes=int(np.round(x[flip]).sum()),
                    scenario_physical_checks=check,
                    emergency_charge_complementarity='scenario-slot binary',
                    common_purchase_bound='max(max_scenario_net,0)+AC_charge_limit',
                    emergency_bound='max(scenario_net,0)')
    return {'purchase':np.maximum(x[q],0.),'allowed_charge':(x[mode]>.5)[np.arange(n)//block_slots],
            'scenario_charge':x[c],'scenario_discharge':x[d],'scenario_emergency':x[e],
            'scenario_surplus':x[w],'scenario_states':states,'metadata':metadata}
