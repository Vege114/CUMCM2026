"""Joint historical error/revision paths with causal, information-based tree nodes."""
from functools import lru_cache

import numpy as np

from .data import protocol


class ScenarioFactory:
    def __init__(self,data,store):
        self.data=data;self.store=store

    @lru_cache(maxsize=128)
    def historical_day(self,day,issued,raw_pv):
        o=day*144;columns=[0,3 if issued else 1,2]
        actual=self.data.actual[o:o+144].copy()
        predictions=np.empty((4,144,3));predictions[:]=np.nan
        for stage in range(4):
            start=stage*36
            p=self.store.get(o+start,issued=issued,raw_pv=raw_pv)
            predictions[stage,start:]=p[:144-start,columns]
        return actual,predictions

    def build(self,origin,forecast,scenario,update_hours=(6,12,18),raw_pv=False,anticipate=True):
        cfg=protocol()["risk"];issued=scenario in ("3","4-3")
        day,start=divmod(int(origin),144); stage=start//36;n=144-start
        # The latest donor day's final 24-hour issue must already be fully realized.
        last_day=(int(origin)-252)//144
        days=list(range(max(1,last_day-cfg["history_days"]+1),last_day+1))
        assert days and max(days)*144+252<=origin
        records=[self.historical_day(d,issued,raw_pv) for d in days]
        truth=np.stack([r[0][start:] for r in records])
        previous=np.stack([r[1][stage,start:] for r in records])
        errors=truth-previous
        mean=errors.mean(0);variance=errors.var(0)
        # Pooled release-specific statistics; PV daytime/nighttime are kept separate.
        pooled_mean=np.tile(mean.mean(0),(n,1));pooled_variance=np.tile(errors.var(axis=(0,1)),(n,1))
        daylight=forecast[:n,3 if issued else 1]>0
        for mask in (daylight,~daylight):
            if mask.any():
                pooled_mean[mask,1]=errors[:,mask,1].mean()
                pooled_variance[mask,1]=errors[:,mask,1].var()
        weight=len(days)/(len(days)+cfg["shrinkage_days"])
        bias=weight*mean+(1-weight)*pooled_mean
        shrunk_variance=weight*variance+(1-weight)*pooled_variance
        ratio=np.sqrt(shrunk_variance/np.maximum(variance,1e-8))
        # Do not amplify a numerically constant group into an invented trajectory.
        ratio=np.where(variance>1e-8,np.minimum(ratio,10),1)
        residual=bias+(errors-mean)*ratio
        indices=np.unique(np.linspace(0,len(days)-1,min(cfg["max_paths"],len(days))).round().astype(int))
        columns=[0,3 if issued else 1,2]
        center=forecast[:n,columns]
        paths=np.maximum(center[None,:,:]+residual[indices],0)
        paths[:,:,2]=np.maximum(paths[:,:,2],1e-4)
        paths[:,~daylight,1]=0
        variable=scenario.startswith("4")
        if not variable:paths[:,:,2]=self.data.fixed_price[start:]
        count=len(indices);groups=np.zeros((count,n),dtype=int)
        boundaries=[h*6-start for h in update_hours if h*6>start] if issued and anticipate else []
        selected_records=[records[i] for i in indices]
        next_group=1;current=np.zeros(count,dtype=int);tree=[]
        for boundary in boundaries:
            future_stage=(start+boundary)//36
            innovations=np.stack([r[1][future_stage,start+boundary:]-r[1][stage,start+boundary:]
                                  for r in selected_records])
            innovations*=ratio[boundary:][None,:,:]
            # Only observed prefix and the forecast released at this node can split it.
            visible=np.concatenate((residual[indices,:boundary].reshape(count,-1),
                                    innovations.reshape(count,-1)),axis=1)
            scales=np.maximum(visible.std(0),1e-6);visible=visible/scales
            updated=current.copy()
            for parent in np.unique(current):
                members=np.flatnonzero(current==parent)
                if len(members)<4:continue  # preserve unresolved terminal disturbances
                x=visible[members];centered=x-x.mean(0)
                if np.max(np.abs(centered))<1e-8:continue
                # First principal component is deterministic up to sign; canonicalize sign.
                gram=centered@centered.T
                eigval,eigvec=np.linalg.eigh(gram);projection=eigvec[:,-1]*np.sqrt(max(eigval[-1],0))
                if projection[np.argmax(np.abs(projection))]<0:projection=-projection
                order=np.argsort(projection,kind="stable");mid=len(order)//2
                if np.isclose(projection[order[mid-1]],projection[order[mid]],atol=1e-9):continue
                for part in (members[order[:mid]],members[order[mid:]]):
                    updated[part]=next_group
                    tree.append({"reveal_slot":start+boundary,"parent":int(parent),
                                 "node":next_group,"members":part.tolist(),
                                 "information":"realized prefix and newly issued forecast revision"})
                    next_group+=1
            current=updated
            groups[:,boundary:]=current[:,None]
        probabilities=np.full(count,1/count)
        return {"paths":paths,"probabilities":probabilities,"groups":groups,
                "boundaries":boundaries,"metadata":{"origin":int(origin),"source_days":days,
                "selected_source_days":[days[i] for i in indices],"paths":count,
                "latest_source_target":max(days)*144+252,"issued_forecast_allowed":issued,
                "shrinkage_weight":weight,"tree":tree,"terminal_uncertainty_preserved":True,
                "bias_mean":bias.mean(0).tolist(),"variance_mean":shrunk_variance.mean(0).tolist()}}
