"""One signed-store bridge to the unchanged tree28/q.8/buffer500 controller."""
import copy
import hashlib
import json
from pathlib import Path

from experiments.exp008.forecast_absolute_hgb import OUT as BASE, array_hash
from experiments.exp008.risk_window import implementation_check, replay
from experiments.exp008.verify import verify_npz
from experiments.problem2.exp003.data import Data


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(store,out,evidence):
    out=Path(out)
    if out.exists():
        raise FileExistsError('Existing bridge evidence is immutable')
    out.mkdir(parents=True)
    paths=[Path(p).resolve() for p in evidence]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(out/'protocol.json',{'model_id':store.name,'days':334,
        'conditioning':'tree','history_days':28,'quantile':.8,'state_buffer':500.,
        'values_sha256':array_hash(store.values),'origins_sha256':array_hash(store.origins),
        'model_evidence_sha256':hashes,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'one_fixed_comparison':True,'development_not_independent_test':True,'final_model_selection':False})
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data=Data()
    save(out/'risk_implementation_verification.json',implementation_check(data,store))
    case=store.name+'_tree28_q08_buffer500'
    replay(case,28,'tree',334,data,store,out)
    directory=out/f'{case}_334days'
    audit=json.loads((directory/'audit.json').read_text());corrected=copy.deepcopy(audit)
    for row in corrected:
        row['forecast_calibration']=store.name
        row['residual_source']='periodic_baseline' if row['fallback'] else store.name+'_same_prior_issued_predictions'
        row['prediction_values_sha256']=array_hash(store.values)
    save(directory/'source_corrected_audit.json',corrected)
    save(directory/'audit_metadata_correction.json',{'original_preserved':'audit.json',
        'corrected_view':'source_corrected_audit.json','numerical_arrays_changed':False,
        'reason':'shared numerical bridge has fixed legacy ridge28 textual labels; source identity follows actual constructor inputs'})
    new=verify_npz(directory/'dispatch_2.npz',audit_path=directory/'source_corrected_audit.json')
    old=verify_npz(BASE/'lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days/dispatch_2.npz',
                   audit_path=BASE/'lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days/audit.json')
    assert new['passed'] and old['passed']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths)
    result={'complete':True,'baseline':old,'candidate':new,
        'cost_change_yuan':new['recomputed_total_cost']-old['recomputed_total_cost'],
        'reversal_change':new['battery_metrics']['direction_reversals']-old['battery_metrics']['direction_reversals'],
        'terminal_soc_difference_kwh':new['battery_metrics']['final_soc']-old['battery_metrics']['final_soc'],
        'all_evidence_unchanged':True,'final_model_selection':False}
    save(out/'paired_findings.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','baseline')},indent=2),flush=True)
    return result
