"""Independent report-only scores from frozen predictions and source actuals."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from experiments.problem2.exp003.data import Data

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/experiments/exp008/evidence/forecast'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run():
    data=Data();actual=data.actual.reshape(365,144,2)[31:365]
    dates=pd.date_range('2025-02-01','2025-12-31')
    sources={};prediction={}
    provenance=json.loads((OUT/'provenance.json').read_text())
    for model,name in [('final_blend','forecast_hgb_extra_trees_half/hgb_extra_trees_half_ridge28_memory.npz'),('single_hgb','forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz')]:
        path=ROOT/'data/results/exp008'/name
        assert sha(path)==provenance['sources_sha256'][str(path.relative_to(ROOT))]
        with np.load(path) as a:
            assert np.array_equal(a['origins'],np.arange(31,365)*144)
            prediction[model]=a['values'].copy()
            np.testing.assert_allclose(a['errors_kw'],actual-a['values'],rtol=0,atol=1e-9)
        sources[model]={'source':str(path.relative_to(ROOT)),'sha256':sha(path)}
    counts={};empty=0
    for kind,expected in [('annual',12),('monthly',132),('lead',1728)]:
        frame=pd.read_csv(OUT/f'{kind}_metrics.csv');assert len(frame)==expected
        for row in frame.to_dict('records'):
            y=actual.copy();f=prediction[row['model']].copy()
            mask=np.ones((334,144),bool) if row['population']=='all' else y[:,:,1]>0
            if kind=='monthly':mask &= np.broadcast_to((dates.month==row['month'])[:,None],mask.shape)
            if kind=='lead':mask &= np.broadcast_to(np.arange(144)[None,:]==row['lead_slot']-1,mask.shape)
            ch={'load':0,'pv':1}.get(row['target'])
            actual_target=y[:,:,ch] if ch is not None else y[:,:,0]-y[:,:,1]
            pred_target=f[:,:,ch] if ch is not None else f[:,:,0]-f[:,:,1]
            errors=(pred_target-actual_target)[mask];truth=actual_target[mask];n=len(errors)
            assert n==row['n'];denom=abs(truth).sum();sae=abs(errors).sum();sse=(errors**2).sum();signed=errors.sum()
            computed=dict(mae_kw=sae/n if n else np.nan,rmse_kw=np.sqrt(sse/n) if n else np.nan,
                wape_pct=100*sae/denom if denom else np.nan,bias_kw=signed/n if n else np.nan,
                absolute_error_sum_kw=sae,squared_error_sum_kw2=sse,signed_error_sum_kw=signed,actual_absolute_sum_kw=denom)
            for metric,value in computed.items():np.testing.assert_allclose(row[metric],value,rtol=1e-11,atol=1e-7,equal_nan=True)
            if not n:
                empty+=1;assert pd.notna(row['missing_reason'])
                assert all(pd.isna(row[k]) for k in ('mae_kw','rmse_kw','wape_pct','bias_kw'))
            elif not denom:assert pd.isna(row['wape_pct']) and pd.notna(row['missing_reason'])
        counts[kind]=dict(rows=len(frame),sha256=sha(OUT/f'{kind}_metrics.csv'))
    result=dict(passed=True,sources=sources,source_actual_hashes=data.hashes,files=counts,
        metrics_recomputed_per_row=8,total_rows=sum(v['rows'] for v in counts.values()),empty_lead_population_rows=empty,
        scope='00:00 forecasts only; all and actualPV>0 for load, PV and net load; no other issue-hour claim',
        no_training_or_optimizer_rerun=True,audit_script_sha256=sha(Path(__file__)))
    (OUT/'independent_report_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=True,rows=result['total_rows'],empty_rows=empty)))
if __name__=='__main__':run()
