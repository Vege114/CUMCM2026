"""Causal physical feedback objective with paired scenario prices.

Dynamics, nonsmooth conventions and optimizer settings reproduce closed_loop;
shared purchases use mean scenario prices, emergency penalties retain each
net path's paired price. Actual target-day prices are never inputs.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

ETA=np.sqrt(.9)
LOW,HIGH,LIMIT=1200.,10800.,5000/6

def objective(purchase, net_paths, scenario_prices, initial_soc, *, charge_mask=None,
              throughput=.005, variation=.001, terminal=.45, deadband=0.,
              emergency_weight=5.):
    paths=np.asarray(net_paths)
    k,n=paths.shape
    prices=np.asarray(scenario_prices,float)
    if prices.shape != (k,n) or not np.isfinite(prices).all() or not np.all(prices > 0):
        raise ValueError('scenario_prices must be positive finite and match net_paths')
    mean_price=(prices[0].copy() if np.all(prices == prices[0]) else prices.mean(axis=0))
    soc=np.full(k,initial_soc)
    sa=np.empty((k,n)); sb=np.empty((k,n))
    local_s=np.zeros((k,n)); local_b=np.zeros((k,n))
    action=np.zeros((k,n)); action_s=np.zeros((k,n));action_b=np.zeros((k,n))
    cost=float(np.dot(purchase,mean_price))
    for t in range(n):
        b=purchase[t]-paths[:,t]
        positive=b>=0
        can_c=True if charge_mask is None else bool(charge_mask[t])
        can_d=True if charge_mask is None else not bool(charge_mask[t])
        available_c=np.maximum(0,(HIGH-soc)/ETA)
        available_d=np.maximum(0,(soc-LOW)*ETA)
        c=np.where(positive&can_c,np.minimum(np.minimum(b,LIMIT),available_c),0.)
        d=np.where((~positive)&can_d,np.minimum(np.minimum(-b,LIMIT),available_d),0.)
        c=np.where(c>=deadband,c,0.)
        cb=(positive&can_c&(b<LIMIT)&(b<available_c)&(c>=deadband)).astype(float)
        cs=np.where(positive&can_c&(available_c<=b)&(available_c<LIMIT)&(c>=deadband),-1/ETA,0.)
        db=-((~positive)&can_d&(-b<LIMIT)&(-b<available_d)).astype(float)
        ds=np.where((~positive)&can_d&(available_d<=-b)&(available_d<LIMIT),ETA,0.)
        e=np.maximum(0,-b-d)
        # e == 0 kink uses the no-emergency side; exact crossings remain
        # nonsmooth and no global convergence certificate is claimed.
        active=e>1e-9
        local_b[:,t]=emergency_weight*prices[:,t]*np.where(active,-1-db,0)+throughput*(cb+db)
        local_s[:,t]=emergency_weight*prices[:,t]*np.where(active,-ds,0)+throughput*(cs+ds)
        sa[:,t]=1+ETA*cs-ds/ETA
        sb[:,t]=ETA*cb-db/ETA
        action[:,t]=c-d
        action_b[:,t]=cb-db
        action_s[:,t]=cs-ds
        soc=soc+ETA*c-d/ETA
        cost+=float(np.mean(emergency_weight*prices[:,t]*e+throughput*(c+d)))
    cost-=float(terminal*np.mean(soc-LOW))
    diff=np.diff(action,axis=1)
    cost+=float(variation*6*np.mean(np.abs(diff).sum(1)))
    tv_grad=np.zeros_like(action)
    tv_grad[:,1:]+=variation*6*np.sign(diff)
    tv_grad[:,:-1]-=variation*6*np.sign(diff)
    local_b+=tv_grad*action_b
    local_s+=tv_grad*action_s
    adj=np.full(k,-terminal)
    grad=mean_price.copy()
    for t in range(n-1,-1,-1):
        grad[t]+=np.mean(local_b[:,t]+adj*sb[:,t])
        adj=local_s[:,t]+adj*sa[:,t]
    return cost,grad

def optimize(initial_purchase, net_paths, scenario_prices, initial_soc, *, charge_mask=None,
             throughput=.005,variation=.001,terminal=.45,deadband=0.,maxiter=100,
             emergency_weight=5.,original=None):
    began=perf_counter()
    prices=np.asarray(scenario_prices)
    n=np.asarray(net_paths).shape[1]
    floor=np.zeros(n) if original is None else np.asarray(original)
    buy_price=prices if original is None else 1.5*prices
    # Pricing of emergency and purchased energy can differ under adjustment.
    # Q2 optimization uses prices for both; adjustment is handled by the joint
    # planner until its own billing-aware closed-loop derivative is implemented.
    if original is not None:
        raise NotImplementedError('Closed-loop adjustment objective is not implemented')
    def fun(g):
        return objective(g,net_paths,buy_price,initial_soc,charge_mask=charge_mask,
                         throughput=throughput,variation=variation,terminal=terminal,
                         deadband=deadband,emergency_weight=emergency_weight)
    result=minimize(fun,np.maximum(initial_purchase,floor),jac=True,method='L-BFGS-B',
                    bounds=list(zip(floor,np.full(n,np.inf))),
                    options={'maxiter':maxiter,'maxls':30,'ftol':1e-9,'gtol':1e-5})
    return {'purchase':result.x,'metadata':{'method':'paired_price_closed_loop_lbfgsb',
            'success':bool(result.success),'message':str(result.message),
            'iterations':int(result.nit),'evaluations':int(result.nfev),
            'objective':float(result.fun),'planning_seconds':perf_counter()-began,
            'global_optimality_certificate':False}}
