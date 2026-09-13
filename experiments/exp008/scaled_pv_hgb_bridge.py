"""Independent retraining audit and one unchanged full-year LP bridge."""
import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_scaled_pv_hgb import OUT, BASE, features, fit, save
from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, ArrayStore, array_hash
from experiments.exp008.risk_window import replay
from experiments.exp008.verify import verify_npz
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.data import split_days


def main():
    directory=OUT/'lp_bridge'
    if directory.exists():
        raise FileExistsError('Existing bridge evidence is immutable')
    directory.mkdir(parents=True)
    previous=json.loads((OUT/'summary.json').read_text())
    assert previous['continuation_gate_passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    with np.load(OUT/'memory.npz') as z:
        store=ArrayStore(z['values'],z['origins'],'scaled_pv_hgb_ridge28_memory_half')
    data=Data();base=AbsoluteHGBStore()
    np.testing.assert_array_equal(store.origins,base.origins)
    save(directory/'protocol.json', {'model_id':store.name,'days':334,
        'spec':{'conditioning':'tree','history_days':28,'quantile':.8,'state_buffer':500},
        'one_fixed_candidate_after_predeclared_forecast_gate':True,
        'values_sha256':array_hash(store.values),'origins_sha256':array_hash(store.origins),
        'forecast_summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'final_model_selection':False,'development_not_independent_test':True})
    (directory/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    checks=[]
    for month in (2,9):
        train,val,formal,cutoff=split_days(month)
        changed=copy.copy(data);changed.actual=data.actual.copy()
        changed.actual[int(formal[0])*144:]+=np.array([40000.,90000.])
        inputs=[features(changed,day) for day in range(7,365)]
        x=np.stack([r[0] for r in inputs]);scales=np.array([r[1] for r in inputs])
        with np.load(OUT/'features.npz') as original:
            np.testing.assert_array_equal(x[np.r_[train,val]-7],original['values'][np.r_[train,val]-7])
            np.testing.assert_array_equal(scales[np.r_[train,val]-7],original['scales'][np.r_[train,val]-7])
        fresh,audit=fit(changed,month,x,scales,protocol['model_configuration'])
        old=joblib.load(OUT/f'pv_m{month:02}.joblib')
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(fresh.predict(x[int(formal[0])-7]),old.predict(x[int(formal[0])-7]))
        checks.append({'month':month,'full_PV_model_retraining_after_future_mutation_identical':True,
                       'training_and_validation_features_and_scales_unchanged':True})
    save(directory/'independent_retraining_verification.json',{'passed':True,'checks':checks})
    result=replay('scaled_pv_hgb_tree28_q08_buffer500',28,'tree',334,data,store,directory)
    case=directory/'scaled_pv_hgb_tree28_q08_buffer500_334days'
    # The shared numerical bridge has legacy textual labels. Preserve its
    # original audit and write an explicitly corrected model-identity view.
    audit=json.loads((case/'audit.json').read_text())
    corrected=copy.deepcopy(audit)
    for row in corrected:
        row['forecast_calibration']=store.name
        row['residual_source']='periodic_baseline' if row['fallback'] else store.name+'_same_prior_issued_predictions'
        row['prediction_values_sha256']=array_hash(store.values)
    save(case/'source_corrected_audit.json',corrected)
    save(case/'audit_metadata_correction.json',{'legacy_source_audit_retained':'audit.json',
        'corrected_view':'source_corrected_audit.json','numerical_arrays_changed':False,
        'reason':'shared WindowResidualScenarios and replay contain fixed legacy ridge28 textual source labels; actual constructor inputs were the signed new store',
        'actual_values_sha256':array_hash(store.values)})
    new=verify_npz(case/'dispatch_2.npz',audit_path=case/'source_corrected_audit.json')
    old=verify_npz(BASE/'lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days/dispatch_2.npz',
                   audit_path=BASE/'lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days/audit.json')
    assert new['passed'] and old['passed']
    comparison={'complete':True,'baseline':old,'candidate':new,
        'cost_change_yuan':new['recomputed_total_cost']-old['recomputed_total_cost'],
        'reversal_change':new['battery_metrics']['direction_reversals']-old['battery_metrics']['direction_reversals'],
        'all_model_retraining_audits_passed':True,'final_model_selection':False}
    save(directory/'paired_findings.json',comparison)
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('candidate','baseline')},indent=2),flush=True)


if __name__=='__main__':
    main()
