"""One physical model, exact causal execution, and independent settlement."""
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ETA = np.sqrt(0.9)
MIN_SOC, MAX_SOC, POWER_ENERGY = 1200.0, 10800.0, 5000/6


class LinearModel:
    def __init__(self):
        self.lower=[];self.upper=[];self.integer=[];self.objective=[]
        self.row=[];self.col=[];self.value=[];self.lb=[];self.ub=[]

    def variables(self, shape, lower=0, upper=np.inf, integer=False, cost=0):
        n=int(np.prod(shape)); start=len(self.lower)
        self.lower.extend(np.broadcast_to(lower,shape).ravel())
        self.upper.extend(np.broadcast_to(upper,shape).ravel())
        self.integer.extend([int(integer)]*n)
        self.objective.extend(np.broadcast_to(cost,shape).ravel())
        return np.arange(start,start+n).reshape(shape)

    def constraint(self, terms, lower=-np.inf, upper=np.inf):
        row=len(self.lb)
        for ids,coefficients in terms:
            ids=np.asarray(ids).ravel(); coefficients=np.broadcast_to(coefficients,ids.shape)
            self.row.extend([row]*len(ids));self.col.extend(ids);self.value.extend(coefficients)
        self.lb.append(lower);self.ub.append(upper)

    def solve(self, seconds=2, gap=.01, fixed=None):
        a=coo_matrix((self.value,(self.row,self.col)),shape=(len(self.lb),len(self.lower))).tocsc()
        a.indices=a.indices.astype(np.int32);a.indptr=a.indptr.astype(np.int32)
        constraints=LinearConstraint(a,np.asarray(self.lb),np.asarray(self.ub))
        began=time.monotonic()
        lower=np.asarray(self.lower).copy();upper=np.asarray(self.upper).copy()
        integrality=np.asarray(self.integer)
        if fixed is not None:
            ids,values=fixed;lower[ids]=values;upper[ids]=values
            integrality=np.zeros_like(integrality)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
            result=milp(np.asarray(self.objective),integrality=integrality,
                        bounds=Bounds(lower,upper),constraints=constraints,
                        options={"time_limit":seconds,"mip_rel_gap":gap,"threads":1})
        valid=False; residual=None
        if result.x is not None and np.isfinite(result.x).all():
            x=result.x;ax=a@x
            residual=max(float(np.max(np.maximum(np.asarray(self.lower)-x,0))),
                         float(np.max(np.maximum(x-np.asarray(self.upper),0))),
                         float(np.max(np.maximum(np.asarray(self.lb)-ax,0))),
                         float(np.max(np.maximum(ax-np.asarray(self.ub),0))))
            ints=np.asarray(self.integer,dtype=bool)
            int_error=float(np.max(np.abs(x[ints]-np.round(x[ints])))) if ints.any() else 0
            valid=residual<1e-5 and int_error<1e-5
        metadata={"status":int(result.status),"message":str(result.message),
                  "seconds":time.monotonic()-began,"feasible":valid,
                  "mip_gap":float(result.mip_gap) if getattr(result,"mip_gap",None) is not None and np.isfinite(result.mip_gap) else None,
                  "node_count":int(getattr(result,"mip_node_count",0) or 0),
                  "constraint_residual":residual,"variables":len(self.lower),
                  "binaries":int(sum(self.integer)),"constraints":len(self.lb)}
        bound=getattr(result,"mip_dual_bound",None)
        metadata["dual_bound"]=float(bound) if bound is not None and np.isfinite(bound) else None
        return result.x if valid else None, metadata


def deterministic(load,pv,price,soc,original=None):
    n=len(load); net=(np.asarray(load)-pv)/6; price=np.maximum(price,1e-6)
    m=LinearModel()
    g=m.variables((n,),upper=np.maximum(net,0)+POWER_ENERGY,cost=price if original is None else 0)
    c=m.variables((n,),upper=POWER_ENERGY,cost=1e-8)
    d=m.variables((n,),upper=POWER_ENERGY,cost=1e-8)
    w=m.variables((n,))
    s=m.variables((n,),lower=MIN_SOC,upper=MAX_SOC)
    z=m.variables((n,),upper=1,integer=True)
    if original is not None:
        up=m.variables((n,),cost=1.5*price); down=m.variables((n,),cost=.5*price)
        for t in range(n):m.constraint([(g[t],1),(up[t],-1),(down[t],1)],original[t],original[t])
    for t in range(n):
        m.constraint([(g[t],1),(c[t],-1),(d[t],1),(w[t],-1)],net[t],net[t])
        terms=[(s[t],1),(c[t],-ETA),(d[t],1/ETA)]
        if t:terms.append((s[t-1],-1))
        rhs=soc if t==0 else 0
        m.constraint(terms,rhs,rhs)
        m.constraint([(c[t],1),(z[t],-POWER_ENERGY)],upper=0)
        m.constraint([(d[t],1),(z[t],POWER_ENERGY)],upper=POWER_ENERGY)
    x,meta=m.solve(seconds=2)
    meta["method"]="deterministic_milp";meta["fallback"]=x is None
    return (np.maximum(x[g],0) if x is not None else np.maximum(net,0)),meta


def execute(grid,load,pv,initial):
    n=len(grid);c,d,e,w=(np.zeros(n) for _ in range(4))
    s=np.empty(n+1);s[0]=initial
    for t in range(n):
        balance=grid[t]+(pv[t]-load[t])/6
        if balance>=0:
            c[t]=min(balance,POWER_ENERGY,max(0,(MAX_SOC-s[t])/ETA));w[t]=balance-c[t]
        else:
            d[t]=min(-balance,POWER_ENERGY,max(0,(s[t]-MIN_SOC)*ETA));e[t]=-balance-d[t]
        s[t+1]=s[t]+ETA*c[t]-d[t]/ETA
    return c,d,e,w,s


def settle(original,final,emergency,price):
    return np.stack((original*price,np.maximum(final-original,0)*1.5*price,
                     np.maximum(original-final,0)*.5*price,emergency*price*5),axis=-1)


def validate_detail(detail):
    d=detail
    for k in ("original","final","charge","discharge","emergency","surplus"):
        assert np.isfinite(d[k]).all() and d[k].min()>=-1e-7
    balance=d["final"]+(d["actual"][:,1]-d["actual"][:,0])/6+d["discharge"]+d["emergency"]-d["charge"]-d["surplus"]
    transition=np.diff(d["states"])-ETA*d["charge"]+d["discharge"]/ETA
    assert max(np.abs(balance).max(),np.abs(transition).max())<1e-6
    assert d["states"].min()>=MIN_SOC-1e-6 and d["states"].max()<=MAX_SOC+1e-6
    assert max(d["charge"].max(),d["discharge"].max())<=POWER_ENERGY+1e-6
    assert not ((d["charge"]>1e-6)&(d["discharge"]>1e-6)).any()
    np.testing.assert_allclose(d["fees"],settle(d["original"],d["final"],d["emergency"],d["price"]),atol=1e-7)
    return float(np.abs(balance).max()),float(np.abs(transition).max())
