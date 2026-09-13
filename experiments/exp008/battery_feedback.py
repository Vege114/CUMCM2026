"""Fixed-purchase stochastic inventory value feedback, with physical projection.

Backward values use available per-slot distributions. This assumes independent
marginal disturbances and is not a perfect-information recourse calculation.
Actual actions inspect only the current observation and precomputed values.
"""
from time import perf_counter
import numpy as np

ETA=np.sqrt(.9)
LOW,HIGH,LIMIT=1200.,10800.,5000/6


def value_functions(purchase, support, price, *, grid_kwh=100., wear=.002,
                    charge_mask=None, terminal=.45, charge_deadband=0.):
    began=perf_counter()
    purchase, support, price=map(np.asarray,(purchase,support,price))
    grid=np.r_[np.arange(LOW,HIGH,grid_kwh),HIGH]
    n=len(price)
    values=np.zeros((n+1,len(grid)))
    values[-1]=-terminal*(grid-LOW)
    delta=grid[None,:]-grid[:,None]
    for t in range(n-1,-1,-1):
        expected=np.zeros(len(grid))
        future=values[t+1]
        for demand in support[t]:
            b=purchase[t]-demand
            positive=b>=0
            permitted=charge_mask is None or bool(charge_mask[t])==positive
            if not permitted:
                expected+=future+5*price[t]*max(-b,0)
                continue
            if positive:
                cmax=np.minimum(np.minimum(b,LIMIT),(HIGH-grid)/ETA)
                cmax=np.where(cmax>=charge_deadband,cmax,0)
                endpoint=grid+ETA*cmax
                eligible=(delta>=-1e-8)&(delta<=ETA*cmax[:,None]+1e-8)
                if charge_deadband:
                    eligible&=(delta<1e-8)|(delta>=ETA*charge_deadband-1e-8)
                costs=future[None,:]+wear*delta/ETA
                end_value=np.interp(endpoint,grid,future)+wear*cmax
                if charge_deadband:
                    minimum=grid+ETA*charge_deadband
                    minimum_value=np.interp(minimum,grid,future)+wear*charge_deadband
                    end_value=np.minimum(end_value,np.where(cmax>=charge_deadband,minimum_value,np.inf))
            else:
                dmax=np.minimum(np.minimum(-b,LIMIT),(grid-LOW)*ETA)
                endpoint=grid-dmax/ETA
                eligible=(delta<=1e-8)&(delta>=-dmax[:,None]/ETA-1e-8)
                costs=future[None,:]+5*price[t]*(-b+ETA*delta)-wear*ETA*delta
                end_value=np.interp(endpoint,grid,future)+5*price[t]*(-b-dmax)+wear*dmax
            grid_value=np.min(np.where(eligible,costs,np.inf),axis=1)
            expected+=np.minimum(grid_value,end_value)
        values[t]=expected/support.shape[1]
    return grid,values,{'method':'fixed_purchase_marginal_value_dp',
                        'seconds':perf_counter()-began,'grid_kwh':grid_kwh,
                        'disturbance_assumption':'independent_marginals',
                        'terminal_value':terminal,'wear':wear}


def execute(purchase, actual, price, initial_soc, grid, values, *, wear=.002,
            charge_mask=None, charge_deadband=0.):
    n=len(purchase)
    c,d,e,w=(np.zeros(n) for _ in range(4))
    states=np.empty(n+1);states[0]=initial_soc
    for t in range(n):
        b=purchase[t]+(actual[t,1]-actual[t,0])/6
        positive=b>=0
        allowed=charge_mask is None or bool(charge_mask[t])==positive
        current=states[t]
        if not allowed:
            following=current
        elif positive:
            full=min(b,LIMIT,max(0,(HIGH-current)/ETA))
            if full<charge_deadband:
                full=0.
            endpoint=current+ETA*full
            choices=np.r_[current,grid[(grid>=current)&(grid<=endpoint)],endpoint]
            if charge_deadband:
                if full>=charge_deadband:
                    choices=np.r_[choices,current+ETA*charge_deadband]
                choices=choices[(choices-current<1e-8)|(choices-current>=ETA*charge_deadband-1e-8)]
            costs=np.interp(choices,grid,values[t+1])+wear*(choices-current)/ETA
            following=choices[np.argmin(costs)]
        else:
            full=min(-b,LIMIT,max(0,(current-LOW)*ETA))
            endpoint=current-full/ETA
            choices=np.r_[current,grid[(grid>=endpoint)&(grid<=current)],endpoint]
            release=(current-choices)*ETA
            costs=np.interp(choices,grid,values[t+1])+5*price[t]*(-b-release)+wear*release
            following=choices[np.argmin(costs)]
        c[t]=max(0,(following-current)/ETA)
        d[t]=max(0,(current-following)*ETA)
        w[t]=max(0,b-c[t]);e[t]=max(0,-b-d[t])
        states[t+1]=following
    fees=np.stack([purchase*price,np.zeros(n),np.zeros(n),5*e*price],axis=-1)
    return dict(original=purchase.copy(),final=purchase.copy(),charge=c,discharge=d,
                emergency=e,surplus=w,states=states,fees=fees,actual=actual.copy(),price=price.copy())
