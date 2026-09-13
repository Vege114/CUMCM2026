"""Past-only local curve trend as a forecasting ablation, no annual fit."""
from pathlib import Path
import json
import numpy as np
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.predict import ForecastStore

OUT=Path(__file__).resolve().parents[2]/'data/results/exp008/trend_forecast'


def predict(actual,day,window=28,half_life=7,degree=1):
    result=np.zeros((144,2))
    dates=np.arange(max(0,day-window),day)
    for channel in (0,1):
        ids=dates[(dates-day)%7==0] if channel==0 else dates
        if not len(ids):
            ids=dates
        x=(ids-day)/7
        design=np.stack([x**i for i in range(degree+1)],axis=1)
        weight=2**((ids-day)/half_life)
        y=np.asarray(actual[:day*144]).reshape(day,144,2)[ids,:,channel]
        penalty=np.diag([0.001]+[.1]*(degree))
        lhs=np.einsum('ni,nj,n->ij',design,design,weight)+penalty
        rhs=np.einsum('ni,nj,n->ij',design,y,weight)
        result[:,channel]=np.linalg.solve(lhs,rhs)[0]
    result=np.maximum(0,result)
    daylight=(np.asarray(actual[max(0,day-28)*144:day*144,1]).reshape(-1,144)>0).any(0)
    result[:,1]*=np.convolve(daylight.astype(int),np.ones(5),mode='same')>0
    return result


def main():
    data=Data();store=ForecastStore('no_season');OUT.mkdir(exist_ok=True,parents=True)
    truth=data.actual.reshape(365,144,2)[31:]
    rows=[]
    for window,half,degree in [(14,7,1),(28,7,1),(42,14,1),(28,14,2)]:
        values=np.stack([predict(data.actual,day,window,half,degree) for day in range(31,365)])
        error=values-truth
        net_error=error[:,:,0]-error[:,:,1]
        name=f'trend_w{window}_h{half}_d{degree}'
        row=dict(name=name,window=window,half_life=half,degree=degree,
                 net_rmse=float(np.sqrt(np.mean(net_error**2))),
                 load_rmse=float(np.sqrt(np.mean(error[:,:,0]**2))),
                 pv_rmse=float(np.sqrt(np.mean(error[:,:,1]**2))),net_bias=float(net_error.mean()),
                 information='all input indices strictly before day*144',role='2025 development ablation')
        np.savez_compressed(OUT/f'{name}.npz',values=values,origins=store.origins,delta=values-store.values)
        rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
