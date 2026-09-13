"""Same fixed Ridge28 and energy memory after the cumulative-loss CNN.

Only the CNN training loss differs from matched Joint/Ridge28/Memory pipelines.
This file does not change or overwrite any of the earlier forecasts.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from experiments.exp008.forecast_calibration import CalibratedStore,CONFIG,_calibrate
from experiments.exp008.load_energy_memory import EnergyMemoryStore
from experiments.exp008.neural_joint_calibration import JointStore,score
from experiments.exp008.forecast import Forecasts
from experiments.problem2.exp003.data import ROOT,Data

OUT=ROOT/'data/results/exp008/cumulative_calibration'
RAW=ROOT/'data/results/exp008/neural_cumulative'


class CumulativeStore:
    def __init__(self,kind='raw'):
        if kind not in ('raw','ridge28','memory'):
            raise ValueError('unknown cumulative pipeline')
        directory=RAW if kind=='raw' else OUT
        filename='predictions.npz' if kind=='raw' else f'cumulative_{kind}.npz'
        meta=json.loads((directory/'manifest.json').read_text())
        if not meta['complete']:
            raise ValueError('Incomplete cumulative prediction archive')
        expected=meta['archive_sha256'] if kind=='raw' else meta['archives'][filename]
        path=directory/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Cumulative archive hash mismatch')
        with np.load(path) as archive:
            self.origins=archive['origins'].copy();self.values=archive['values'].copy()
        self.name=f'cumulative_cnn_{kind}'
        self.lookup={int(origin):i for i,origin in enumerate(self.origins)}

    def get(self,origin):
        return self.values[self.lookup[int(origin)]].copy()


class CumulativeForecasts(Forecasts):
    def __init__(self,kind='memory',data=None):
        super().__init__(data=data)
        self.store=CumulativeStore(kind)
        self.calibration=self.store.name

    def get(self,day,slot=0,scenario='2'):
        result=super().get(day,slot,scenario)
        result['audit'].update(
            base_forecast='exp008_shared_cnn_cumulative_loss' if day>=31 else 'periodic_cold_start',
            output_calibration=self.calibration if day>=31 else None,
            parameter_count=4850 if day>=31 else None,
            base_architecture_unchanged=day<31,
            joint_architecture_unchanged=True,
            extra_training_objective='normalized cumulative net energy Huber',
            all_correction_labels_before_issue=True)
        return result


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise FileExistsError('Completed cumulative comparison is immutable')
    data=Data();raw=CumulativeStore()
    protocol={'training_candidate_count':1,'calibrator':CONFIG['ridge_28'],
              'memory_gain':.5,'comparison':'same Ridge28 and same memory before/after CNN loss change',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'raw_values_sha256':hashlib.sha256(raw.values.tobytes()).hexdigest(),
              'gate_for_lp_bridge':'daily energy, prefix energy and high-price RMSE all beat Joint+Ridge28+Memory',
              'development_on_previously_examined_2025':True}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    calibrated=CalibratedStore('ridge_28',seed=42,data=data,directory=OUT/'calibration',
                               use_cache=False,_base_store=raw)
    memory=EnergyMemoryStore(data=data,base_store=calibrated)
    for item in memory.audit:
        item['underlying_forecast']='cumulative_loss_CNN_then_fixed_Ridge28'
    archives={}
    for name,store in [('ridge28',calibrated),('memory',memory)]:
        path=OUT/f'cumulative_{name}.npz'
        np.savez_compressed(path,origins=store.origins,values=store.values)
        archives[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (OUT/f'{name}_audit.json').write_text(json.dumps(store.audit,indent=2))
    pollution=[]
    for day in (31,62,181,243,364):
        index=raw.lookup[day*144]
        clean,_=_calibrate(data.actual,raw.origins,raw.values,index,CONFIG['ridge_28'])
        changed=data.actual.copy();changed[day*144:]+=70000.
        altered,_=_calibrate(changed,raw.origins,raw.values,index,CONFIG['ridge_28'])
        error=float(np.max(np.abs(clean-altered)))
        assert error==0.
        pollution.append({'day':day,'calibration_future_mutation_error':error})
    truth=data.actual[raw.origins[:,None]+np.arange(144)]
    joint=JointStore(calibrated=True)
    reference_memory=EnergyMemoryStore(data=data,base_store=joint)
    pipelines=[('joint_ridge28',joint),('joint_memory',reference_memory),('cumulative_raw',raw),
               ('cumulative_ridge28',calibrated),('cumulative_memory',memory)]
    annual=[];monthly=[]
    dates=pd.date_range('2025-02-01','2025-12-31')
    for name,store in pipelines:
        annual.append({'name':name,**score(store.values,truth,data.fixed_price)})
        for month in range(2,13):
            ids=dates.month==month
            monthly.append({'name':name,'month':month,**score(store.values[ids],truth[ids],data.fixed_price)})
    pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    metrics={row['name']:row for row in annual}
    gate_metrics=('daily_net_energy_rmse_kwh','intraday_cumulative_error_rmse_kwh','high_price_net_rmse_kw')
    gate=all(metrics['cumulative_memory'][key]<metrics['joint_memory'][key] for key in gate_metrics)
    manifest={'complete':True,'archives':archives,'metrics':annual,'lp_bridge_gate_passed':gate,
              'future_mutation_checks':pollution,'all_ridge_labels_before_issue':all(
                  item['history_last_label'] is None or item['history_last_label']<item['origin']
                  for item in calibrated.audit),
              'pv_memory_layer_unchanged':bool(np.array_equal(memory.values[:,:,1],calibrated.values[:,:,1]))}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    run()
