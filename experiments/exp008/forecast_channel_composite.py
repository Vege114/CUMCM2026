"""One fixed raw-channel composition, followed by identical causal calibration.

Load uses the monthly best-validation-stage HGB; PV uses the past-scale HGB.
This combination is a development decision after observing component metrics,
not an untouched test and not a daily choice based on realized outcome.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, ArrayStore, array_hash
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, CONFIG
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.audited_store_bridge import run as bridge, save
from experiments.problem2.exp003.data import Data

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/exp008/forecast_channel_composite'
LOAD=ROOT/'data/results/exp008/hgb_stage_selection'
PV=ROOT/'data/results/exp008/forecast_scaled_pv_hgb'


def main():
    if OUT.exists():
        raise FileExistsError('Existing evidence is immutable')
    OUT.mkdir(parents=True)
    inputs=[LOAD/'raw.npz',PV/'raw.npz',LOAD/'training_audit.json',PV/'training_audit.json',
            LOAD/'independent_retraining_verification.json',PV/'lp_bridge/independent_retraining_verification.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    save(OUT/'protocol.json',{'load_channel':'monthly best-validation-stage absolute HGB',
        'PV_channel':'past-positive-PV-p95-normalized HGB','fixed_channel_choices_all334days':True,
        'postprocessing':'same Ridge28 and .5 nonrecursive load memory refitted on combined raw values',
        'input_sha256':hashes,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'one_fixed_candidate_no_weight_or_channel_search':True,
        'designed_after_examining_component_development_metrics':True,
        'continuation_gate':'all four net/high-price/daily-energy/prefix RMSE improve on original HGB complete pipeline before a fixed334 LP bridge',
        'untouched_test':False,'final_model_selection':False})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    with np.load(LOAD/'raw.npz') as a,np.load(PV/'raw.npz') as b:
        np.testing.assert_array_equal(a['origins'],b['origins'])
        values=np.stack((a['values'][:,:,0],b['values'][:,:,1]),axis=-1)
        origins=a['origins'].copy()
        np.testing.assert_array_equal(values[:,:,0],a['values'][:,:,0])
        np.testing.assert_array_equal(values[:,:,1],b['values'][:,:,1])
    data=Data();raw=ArrayStore(values,origins,'channel_composite_raw')
    ridge=CalibratedStore('ridge_28',data=data,use_cache=False,_base_store=raw,directory=OUT/'calibration')
    memory=EnergyMemoryStore(data=data,base_store=ridge)
    memory.name='best_stage_load_scaled_pv_ridge28_memory'
    for row in memory.audit:
        row['underlying_forecast']='fixed_raw_channel_composite_then_Ridge28'
    for name,store in [('raw',raw),('ridge',ridge),('memory',memory)]:
        np.savez_compressed(OUT/f'{name}.npz',values=store.values,origins=store.origins)
    save(OUT/'ridge_audit.json',ridge.audit);save(OUT/'memory_audit.json',memory.audit)
    checks=[]
    for i in (0,1,28,59,120,240,333):
        changed=data.actual.copy();changed[int(origins[i]):]+=[60000.,30000.]
        future=values.copy();future[i+1:]+=80000.
        a,_=_calibrate(data.actual,origins,values,i,CONFIG['ridge_28'])
        b,_=_calibrate(changed,origins,future,i,CONFIG['ridge_28'])
        np.testing.assert_array_equal(a,b)
        future=ridge.values.copy();future[i+1:]+=90000.
        a,_=correct_day(data.actual,origins,ridge.values,i)
        b,_=correct_day(changed,origins,future,i)
        np.testing.assert_array_equal(a,b)
        checks.append({'day':int(origins[i]//144),'future_actual_and_raw_calibrated_forecast_mutations_unchanged':True})
    truth=data.actual[origins[:,None]+np.arange(144)]
    metrics=[]
    for name,v in [('original_HGB',AbsoluteHGBStore().values),('channel_composite',memory.values)]:
        for channel,e in [('load',v[:,:,0]-truth[:,:,0]),('pv',v[:,:,1]-truth[:,:,1]),
                         ('net',(v[:,:,0]-v[:,:,1])-(truth[:,:,0]-truth[:,:,1]))]:
            metrics.append({'name':name,'channel':channel,**score(e,data.fixed_price)})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    old,new=[r for r in metrics if r['channel']=='net']
    gate=all(new[k]<old[k] for k in ['rmse_kw','high_price_rmse_kw','daily_energy_rmse_kwh','cumulative_error_rmse_kwh'])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in inputs)
    save(OUT/'summary.json',{'complete':True,'metrics':metrics,'continuation_gate_passed':gate,
        'postprocessing_causality_checks':checks,'raw_channels_match_signed_sources':True,
        'raw_training_causality_inherited_from_verified_source_models':True,
        'input_sources_unchanged':True,'values_sha256':array_hash(memory.values)})
    print(json.dumps({'net_metrics':[old,new],'gate':gate},indent=2),flush=True)
    if gate:
        bridge(memory,OUT/'lp_bridge',[OUT/'summary.json',OUT/'protocol.json',OUT/'memory.npz'])


if __name__=='__main__':
    main()
