"""Independent monthly stage-selection retraining checks then fixed bridge."""
import copy
import json

import joblib
import numpy as np

from experiments.exp008.hgb_stage_selection import OUT, BASE, best_stage, outputs, save
from experiments.exp008.forecast_absolute_hgb import ArrayStore, features_for_day, fit_month
from experiments.exp008.audited_store_bridge import run
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.data import split_days


def main():
    assert json.loads((OUT/'summary.json').read_text())['continuation_gate_passed']
    protocol=json.loads((BASE/'protocol.json').read_text())
    protocol['model_configuration']['max_iter']=400
    data=Data(); checks=[]
    for month in (2,9):
        train,val,formal,cutoff=split_days(month)
        changed=copy.copy(data);changed.actual=data.actual.copy()
        changed.actual[int(formal[0])*144:]+=[30000.,70000.]
        x=np.stack([features_for_day(changed,d)[0] for d in range(7,365)])
        models,audit=fit_month(changed,month,x,protocol)
        stages=[best_stage(m) for m in models]
        old=joblib.load(OUT/f'm{month:02}.joblib')
        assert stages==old['stages']
        np.testing.assert_array_equal(outputs(changed,int(formal[0]),models,stages),
                                      outputs(data,int(formal[0]),old['models'],old['stages']))
        old_audit=next(a for a in json.loads((OUT/'training_audit.json').read_text()) if a['month']==month)
        for key in ('training_feature_hash','validation_feature_hash','training_label_hash','validation_label_hash'):
            assert audit[key]==old_audit[key]
        checks.append({'month':month,'full_model_retraining_and_validation_stage_unchanged':True,
                       'all_train_validation_feature_target_hashes_identical':True})
    save(OUT/'independent_retraining_verification.json',{'passed':True,'checks':checks})
    with np.load(OUT/'memory.npz') as z:
        store=ArrayStore(z['values'],z['origins'],'best_validation_stage_hgb_ridge28_memory')
    run(store,OUT/'lp_bridge',[OUT/'summary.json',OUT/'protocol.json',OUT/'memory.npz',
        OUT/'training_audit.json',OUT/'independent_retraining_verification.json'])


if __name__=='__main__':
    main()
