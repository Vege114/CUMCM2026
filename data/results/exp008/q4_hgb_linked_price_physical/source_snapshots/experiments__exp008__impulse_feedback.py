"""Inventory value control with an explicit last-direction switching charge.

Adapted idea: compare an action's complete continuation value with waiting.
The auxiliary switching charge is never included in the competition bill.
"""
from time import perf_counter
import numpy as np

ETA=np.sqrt(.9)
LOW,HIGH,LIMIT=1200.,10800.,5000/6
MODES=(-1,0,1)


def value_functions(purchase,support,price,*,grid_kwh=100.,wear=.002,
                    switching=50.,terminal=.45):
    began=perf_counter()
    grid=np.r_[np.arange(LOW,HIGH,grid_kwh),HIGH]
    n=len(price)
    values=np.zeros((n+1,3,len(grid)))
    values[-1]=-terminal*(grid-LOW)[None,:]
    delta=grid[None,:]-grid[:,None]
    for t in range(n-1,-1,-1):
        expected=np.zeros((3,len(grid)))
        future=values[t+1]
        for demand in support[t]:
            b=purchase[t]-demand
            positive=b>=0
            direction=1 if positive else -1
            nextv=future[direction+1]
            if positive:
                available=np.minimum(np.minimum(b,LIMIT),(HIGH-grid)/ETA)
                endpoint=grid+ETA*available
                eligible=(delta>1e-8)&(delta<=ETA*available[:,None]+1e-8)
                costs=nextv[None,:]+wear*delta/ETA
                end_value=np.interp(endpoint,grid,nextv)+wear*available
            else:
                available=np.minimum(np.minimum(-b,LIMIT),(grid-LOW)*ETA)
                endpoint=grid-available/ETA
                eligible=(delta<-1e-8)&(delta>=-available[:,None]/ETA-1e-8)
                costs=nextv[None,:]+5*price[t]*(-b+ETA*delta)-wear*ETA*delta
                end_value=np.interp(endpoint,grid,nextv)+5*price[t]*(-b-available)+wear*available
            active=np.minimum(np.min(np.where(eligible,costs,np.inf),axis=1),
                              np.where(available>1e-8,end_value,np.inf))
            for mi,mode in enumerate(MODES):
                idle=future[mi]+5*price[t]*max(-b,0)
                changed=active+(switching if mode*direction==-1 else 0)
                expected[mi]+=np.minimum(idle,changed)
        values[t]=expected/support.shape[1]
    return grid,values,{'method':'inventory_value_with_last_direction',
                       'seconds':perf_counter()-began,'switching_yuan_auxiliary':switching,
                       'grid_kwh':grid_kwh,'terminal':terminal,
                       'disturbance_assumption':'independent_conditional_marginals'}


def execute(purchase,actual,price,initial_soc,grid,values,*,wear=.002,switching=50.,initial_mode=1):
    n=len(purchase)
    c,d,e,w=(np.zeros(n) for _ in range(4))
    states=np.empty(n+1);states[0]=initial_soc
    mode=initial_mode
    for t in range(n):
        b=purchase[t]+(actual[t,1]-actual[t,0])/6
        current=states[t]
        positive=b>=0
        direction=1 if positive else -1
        future=values[t+1]
        idle_value=np.interp(current,grid,future[mode+1])+5*price[t]*max(-b,0)
        if positive:
            available=min(b,LIMIT,max(0,(HIGH-current)/ETA))
            endpoint=current+ETA*available
            choices=np.r_[grid[(grid>current)&(grid<=endpoint)],endpoint]
            costs=np.interp(choices,grid,future[2])+wear*(choices-current)/ETA
        else:
            available=min(-b,LIMIT,max(0,(current-LOW)*ETA))
            endpoint=current-available/ETA
            choices=np.r_[grid[(grid>=endpoint)&(grid<current)],endpoint]
            release=(current-choices)*ETA
            costs=np.interp(choices,grid,future[0])+5*price[t]*(-b-release)+wear*release
        if mode*direction==-1:
            costs+=switching
        best=int(np.argmin(costs))
        following=float(choices[best]) if available>1e-8 and costs[best]<idle_value-1e-8 else current
        c[t]=max(0,(following-current)/ETA);d[t]=max(0,(current-following)*ETA)
        if c[t]+d[t]>1e-8:
            mode=direction
        e[t]=max(0,-b-d[t]);w[t]=max(0,b-c[t]);states[t+1]=following
    fees=np.stack([purchase*price,np.zeros(n),np.zeros(n),5*e*price],axis=-1)
    return dict(original=purchase.copy(),final=purchase.copy(),charge=c,discharge=d,
                emergency=e,surplus=w,states=states,fees=fees,actual=actual.copy(),price=price.copy()),mode
