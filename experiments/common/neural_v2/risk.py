"""Multistage MILP with an exact piecewise encoding of the real causal executor."""
import time

import numpy as np

from .physics import ETA,MAX_SOC,MIN_SOC,POWER_ENERGY,LinearModel,deterministic,execute


def weighted_cvar(costs,probabilities,alpha=.9):
    order=np.argsort(costs)[::-1];remaining=1-alpha;total=0.
    for i in order:
        amount=min(remaining,float(probabilities[i]));total+=amount*costs[i];remaining-=amount
        if remaining<=1e-12:break
    return float(total/(1-alpha))


def optimize(bundle,soc,forecast,weight=0,original=None,adjustable=True,time_limit=2):
    began=time.monotonic();paths=bundle["paths"];prob=bundle["probabilities"]
    groups=bundle["groups"];count,n,_=paths.shape
    net=(paths[:,:,0]-paths[:,:,1])/6;prices=paths[:,:,2]
    upper=np.maximum(np.maximum(net.max(0),(forecast[:,0]-forecast[:,1])/6),0)+POWER_ENERGY
    if original is not None:upper=np.maximum(upper,original)
    model=LinearModel()
    g=model.variables((n,),upper=upper) if original is None else None
    grid=np.empty((count,n),dtype=int)
    # Variables shared by all paths having the same observed information history.
    for t in range(n):
        for node in np.unique(groups[:,t]):
            members=np.flatnonzero(groups[:,t]==node)
            if original is None and (not adjustable or node==0):idx=int(g[t])
            else:idx=int(model.variables((1,),upper=upper[t])[0])
            grid[members,t]=idx
    c=model.variables((count,n),upper=POWER_ENERGY)
    d=model.variables((count,n),upper=POWER_ENERGY)
    e=model.variables((count,n),upper=np.maximum(net,0))
    waste_bound=np.maximum(upper[None,:]-net,0)
    w=model.variables((count,n),upper=waste_bound)
    s=model.variables((count,n),lower=MIN_SOC,upper=MAX_SOC)
    mode=model.variables((count,n),upper=1,integer=True)
    # Two selectors per direction; the remaining selector is their complement.
    selector=model.variables((count,n,4),upper=1,integer=True)
    if adjustable:
        up=model.variables((count,n));down=model.variables((count,n))
    cost=model.variables((count,),cost=prob)
    if weight:
        zeta=model.variables((1,),cost=weight)
        excess=model.variables((count,),cost=weight*prob/.1)
    capacity=MAX_SOC-MIN_SOC
    for k in range(count):
        for t in range(n):
            q=grid[k,t];a,b,f,h=selector[k,t];z=mode[k,t]
            model.constraint([(q,1),(c[k,t],-1),(d[k,t],1),(e[k,t],1),(w[k,t],-1)],net[k,t],net[k,t])
            terms=[(s[k,t],1),(c[k,t],-ETA),(d[k,t],1/ETA)]
            if t:terms.append((s[k,t-1],-1))
            rhs=soc if t==0 else 0
            model.constraint(terms,rhs,rhs)
            model.constraint([(c[k,t],1),(z,-POWER_ENERGY)],upper=0)
            model.constraint([(d[k,t],1),(z,POWER_ENERGY)],upper=POWER_ENERGY)
            model.constraint([(w[k,t],1),(z,-waste_bound[k,t])],upper=0)
            urgent_bound=max(net[k,t],0)
            model.constraint([(e[k,t],1),(z,urgent_bound)],upper=urgent_bound)
            model.constraint([(a,1),(b,1),(z,-1)],upper=0)
            model.constraint([(f,1),(h,1),(z,1)],upper=1)
            # Positive: waste=0 OR charge hits power OR next SOC hits capacity.
            model.constraint([(w[k,t],1),(a,waste_bound[k,t])],upper=waste_bound[k,t])
            model.constraint([(c[k,t],1),(b,-POWER_ENERGY)],lower=0)
            model.constraint([(s[k,t],1),(z,-capacity),(a,capacity),(b,capacity)],lower=MIN_SOC)
            # Negative: emergency=0 OR discharge hits power OR SOC hits lower bound.
            model.constraint([(e[k,t],1),(f,urgent_bound)],upper=urgent_bound)
            model.constraint([(d[k,t],1),(h,-POWER_ENERGY)],lower=0)
            model.constraint([(s[k,t],1),(z,-capacity),(f,-capacity),(h,-capacity)],upper=MIN_SOC)
            if adjustable:
                terms=[(q,1),(up[k,t],-1),(down[k,t],1)]
                if g is not None:terms.append((g[t],-1))
                rhs=float(original[t]) if original is not None else 0
                model.constraint(terms,rhs,rhs)
        terms=[(cost[k],1),(e[k],-5*prices[k])]
        if g is not None:terms.append((g,-prices[k]))
        if adjustable:terms.extend([(up[k],-1.5*prices[k]),(down[k],-.5*prices[k])])
        constant=float(np.dot(original,prices[k])) if original is not None else 0
        model.constraint(terms,constant,constant)
        if weight:model.constraint([(cost[k],1),(zeta[0],-1),(excess[k],-1)],upper=0)
    assembly=time.monotonic()-began
    # Produce a feasible incumbent in an exact controller region before the bounded MIP search.
    # These binaries come from executing an information-admissible common plan, not foresight.
    if original is None:
        initial_plan,initial_meta=deterministic(forecast[:,0],forecast[:,1],forecast[:,2],soc)
    else:
        initial_plan=original.copy();initial_meta=None
    modes=np.zeros((count,n));selectors=np.zeros((count,n,4))
    for k in range(count):
        cc,dd,ee,ww,ss=execute(initial_plan,paths[k,:,0],paths[k,:,1],soc)
        balance=initial_plan-net[k]
        modes[k]=balance>=0
        for t in range(n):
            if balance[t]>=0:
                if ww[t]<1e-7:selectors[k,t,0]=1
                elif abs(cc[t]-POWER_ENERGY)<1e-7:selectors[k,t,1]=1
            else:
                if ee[t]<1e-7:selectors[k,t,2]=1
                elif abs(dd[t]-POWER_ENERGY)<1e-7:selectors[k,t,3]=1
    fixed=(np.r_[mode.ravel(),selector.ravel()],np.r_[modes.ravel(),selectors.ravel()])
    incumbent,incumbent_meta=model.solve(seconds=min(.5,time_limit/2),fixed=fixed)
    if incumbent is not None:
        value=float(np.dot(model.objective,incumbent))
        active=np.flatnonzero(model.objective)
        model.constraint([(active,np.asarray(model.objective)[active])],upper=value+1e-6)
    remaining=max(.05,time_limit-incumbent_meta["seconds"])
    x,meta=model.solve(seconds=remaining)
    if incumbent is not None and (x is None or np.dot(model.objective,incumbent)<np.dot(model.objective,x)):
        x=incumbent
        meta["feasible"]=True;meta["feasible_origin"]="fixed_controller_region_incumbent"
        meta["constraint_residual"]=incumbent_meta["constraint_residual"]
        bound=meta["dual_bound"]
        value=float(np.dot(model.objective,x))
        meta["mip_gap"]=max(0,(value-bound)/max(abs(value),1e-9)) if bound is not None else None
    else:meta["feasible_origin"]="mip_search" if x is not None else None
    meta["incumbent_solver"]=incumbent_meta
    meta["initial_plan_solver"]=initial_meta
    meta["solver_seconds"]=meta["seconds"]+incumbent_meta["seconds"]
    meta.update(method="multistage_exact_greedy_milp",weight=float(weight),paths=count,
                assembly_seconds=assembly,origin=bundle.get("metadata",{}).get("origin"),
                anticipate=bool(bundle["boundaries"]),fallback=x is None)
    if x is not None:
        controller_error=0.
        for k in range(count):
            actual=execute(x[grid[k]],paths[k,:,0],paths[k,:,1],soc)
            for calculated,indices in zip(actual,(c[k],d[k],e[k],w[k],s[k])):
                if len(calculated)==n+1:calculated=calculated[1:]
                controller_error=max(controller_error,float(np.max(np.abs(calculated-x[indices]))))
        meta["controller_error"]=controller_error
        if controller_error>1e-4:
            x=None;meta["fallback"]=True;meta["message"]+="; independent controller mismatch"
    if x is None:
        plan,fallback=deterministic(forecast[:,0],forecast[:,1],forecast[:,2],soc,original)
        meta["fallback_solver"]=fallback
    else:
        costs=x[cost]
        meta.update(expected_cost=float(prob@costs),cvar90=weighted_cvar(costs,prob),
                    objective=float(prob@costs+weight*weighted_cvar(costs,prob)))
        if original is None:plan=np.maximum(x[g],0)
        else:
            plan=original.copy()
            first_stop=min(bundle["boundaries"]) if bundle["boundaries"] else n
            plan[:first_stop]=np.maximum(x[grid[0,:first_stop]],0)
    meta["total_seconds"]=time.monotonic()-began
    return plan,meta
