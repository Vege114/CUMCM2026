"""Proper-score diagnostic for conditional marginals with historical ranks.

Whole historical day ranks preserve an empirical copula; marginal supports
come from the same forecast-conditioned tree already used by the cheap LP.
This is not a full nonanticipative scenario tree or a new prediction network.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.exp008.load_energy_memory import EnergyMemoryForecasts
from experiments.problem2.tree_planning.risk import TreeResidualScenarios,QUANTILE_LEVELS

OUT=Path('data/results/exp008/conditional_path_diagnostic')


def conditional_paths(raw_paths,support):
    raw_paths,support=np.asarray(raw_paths),np.asarray(support)
    ranks=(rankdata(raw_paths,axis=0,method='average')-.5)/len(raw_paths)
    return np.stack([np.interp(ranks[:,t],QUANTILE_LEVELS,support[t])
                     for t in range(raw_paths.shape[1])],axis=1)


def crps(samples,truth):
    """Empirical distribution CRPS, including diagonal pairs (V statistic)."""
    samples,truth=np.asarray(samples),np.asarray(truth)
    return np.abs(samples-truth).mean(axis=0)-.5*np.abs(samples[:,None]-samples[None,:]).mean(axis=(0,1))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    forecast=EnergyMemoryForecasts()
    risk=TreeResidualScenarios(forecast.store.origins,forecast.store.values,
        forecast.data.actual[:,:2],forecast.data.fixed_price)
    protocol={'candidate':'tree conditional marginals with raw historical daily rank copula',
              'reference':'raw 28 complete same-forecast error paths',
              'forecast':'fixed Joint+Ridge28+0.5memory','fit_parameters_changed':False,
              'scoring':'empirical V-statistic CRPS of interval net energy and cumulative prefix energy',
              'prediction_truth_read_only_after_paths_constructed':True,
              'gate':'price weighted slot CRPS and cumulative prefix CRPS both improve',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];audits=[]
    for day in range(31,365):
        issue=forecast.get(day,scenario='2')
        history=forecast.net_error_paths(day,'2',limit=28)
        net=(issue['load_kw']-issue['pv_kw'])/6
        raw=net[None,:]+history['errors_kwh']
        support,audit=risk.for_day(day)
        conditional=conditional_paths(raw,support)
        observed=forecast.data.actual[day*144:(day+1)*144]
        truth=(observed[:,0]-observed[:,1])/6
        for name,paths in [('raw',raw),('conditional',conditional)]:
            score=crps(paths,truth)
            prefix=crps(np.cumsum(paths,axis=1),np.cumsum(truth))
            low,high=np.quantile(paths,[.1,.9],axis=0)
            rows.append({'day':day,'name':name,'slot_crps_kwh':float(score.mean()),
                'price_weighted_slot_crps':float(np.average(score,weights=issue['price'])),
                'prefix_crps_kwh':float(prefix.mean()),'daily_energy_crps_kwh':float(prefix[-1]),
                'central80_coverage':float(np.mean((truth>=low)&(truth<=high))),
                'quantile80_coverage':float(np.mean(truth<=np.quantile(paths,.8,axis=0)))})
        audits.append({'day':day,'history':history['audit'],'tree':audit})
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'daily.csv',index=False)
    means=frame.drop(columns='day').groupby('name').mean()
    means.to_csv(OUT/'annual_scores.csv')
    passed=bool(means.loc['conditional','price_weighted_slot_crps']<means.loc['raw','price_weighted_slot_crps']
        and means.loc['conditional','prefix_crps_kwh']<means.loc['raw','prefix_crps_kwh'])
    summary={'metrics':means.to_dict('index'),'gate_passed':passed,
             'all_labels_before_issue':all(a['tree']['max_observed_index']<a['day']*144 for a in audits),
             'role':'2025 development risk diagnostic; not realized purchase bills'}
    (OUT/'audit.json').write_text(json.dumps(audits,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
