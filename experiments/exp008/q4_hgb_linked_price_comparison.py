"""Inspect completed controlled Q4-2 price-transfer results; no report graphics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import verify_npz

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/exp008/q4_hgb_linked_price_physical'


def main():
    target=OUT/'comparison.json'
    if target.exists():
        raise FileExistsError('Completed comparison is immutable')
    assert json.loads((OUT/'full334/independent_audit.json').read_text())['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    names={
        'prior_ridge28_original_price':'q4_physical',
        'absolute_HGB_original_price':'q4_absolute_hgb_physical',
        'absolute_HGB_linked_price':'q4_hgb_linked_price_physical'}
    summaries,verifications,daily,monthly=[],{},{},[]
    arrays={}
    for label,folder in names.items():
        path=ROOT/'data/results/exp008'/folder/'full334/dispatch_4-2.npz'
        check=verify_npz(path,scenario='4-2',expected_days=334,
            initial_soc=protocol['initial_soc'],initial_mode=protocol['initial_mode'],
            initial_power_kw=protocol['initial_power_kw'])
        assert check['passed'],check['errors']
        verifications[label]=check
        with np.load(path,allow_pickle=False) as pack:
            data={k:pack[k].copy() for k in pack.files}
        arrays[label]=data
        battery=check['battery_metrics']
        summaries.append({'label':label,**check['billing'],**battery,
            'total_charge_discharge_episodes':battery['charge_starts']+battery['discharge_starts'],
            'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        d=pd.DataFrame({'day':np.arange(31,365),
            'month':pd.date_range('2025-02-01','2025-12-31').month,
            'planned_cost':data['fees'][...,0].sum(axis=1),
            'emergency_cost':data['fees'][...,3].sum(axis=1),
            'total_cost':data['fees'].sum(axis=(1,2)),
            'final_soc':data['states'][:,-1]})
        daily[label]=d
        for month,group in d.groupby('month'):
            monthly.append({'label':label,'month':int(month),
                **{k:float(group[k].sum()) for k in ('planned_cost','emergency_cost','total_cost')}})
    frame=pd.DataFrame(summaries)
    frame.to_csv(OUT/'three_layer_comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_three_layer_comparison.csv',index=False)
    new=frame.iloc[-1]; old=frame.iloc[-2]
    # Only the immediate previous HGB group isolates the price model change.
    metrics=('total_cost','planned_cost','emergency_cost','direction_reversals','active_slots',
        'charge_starts','discharge_starts','total_charge_discharge_episodes','throughput_kwh',
        'equivalent_full_cycles','power_ramp_total_kw','final_soc')
    delta={k:float(new[k]-old[k]) for k in metrics}
    assert all(np.array_equal(arrays['absolute_HGB_linked_price'][k],arrays['absolute_HGB_original_price'][k])
               for k in ('actual','price','days'))
    actual=arrays['absolute_HGB_linked_price']
    final_day_audit=json.loads((OUT/'full334/audit.json').read_text())
    gaps=[r['mip']['mip_gap'] for r in final_day_audit if r['mip']['mip_gap'] is not None]
    result={'complete_334day':True,'source_actual_prices_identical':True,
        'three_groups':summaries,'controlled_delta_linked_price_minus_previous_HGB':delta,
        'actual_fee_reduction_vs_previous_HGB_pct':float(100*(1-new.total_cost/old.total_cost)),
        'both_actual_fee_and_reversals_reduced':bool(new.total_cost<old.total_cost and new.direction_reversals<old.direction_reversals),
        'both_actual_fee_and_total_episodes_reduced':bool(new.total_cost<old.total_cost and new.total_charge_discharge_episodes<old.total_charge_discharge_episodes),
        'mip_all334_feasible':all(r['mip']['feasible'] for r in final_day_audit),
        'maximum_mip_gap':max(gaps),'mean_mip_gap':float(np.mean(gaps)),
        'refinement_success_days':sum(r['refinement']['success'] for r in final_day_audit),
        'causal_price_vs_realized_price':True,'initial_soc':float(actual['states'][0,0]),
        'this_is_not_a_Q2_goal_test':True,'no_battery_lifetime_inference':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Preserve signed raw audit records; correct only an inherited display label
    # in a separate view, after numerical HGB-source equality is established.
    raw_audit=OUT/'full334/audit.json'
    original_hash=hashlib.sha256(raw_audit.read_bytes()).hexdigest()
    corrected=json.loads(raw_audit.read_text())
    for row in corrected:
        meta=row['forecast']
        assert meta['base_forecast']=='direct_absolute_load_pv_HGB'
        assert meta['selected_model_id']=='direct_hgb_ridge28_memory_half'
        assert meta['load_method']=='common_midnight_cnn_remaining_trajectory'
        meta['load_method']='common_midnight_direct_absolute_HGB_Ridge28_memory_half_remaining_trajectory'
    corrected_path=OUT/'full334/source_corrected_audit.json'
    corrected_path.write_text(json.dumps(corrected,ensure_ascii=False,indent=2)+'\n')
    assert hashlib.sha256(raw_audit.read_bytes()).hexdigest()==original_hash
    erratum={'metadata_only':True,'original_audit_sha256':original_hash,
        'corrected_audit_sha256':hashlib.sha256(corrected_path.read_bytes()).hexdigest(),
        'changed_field':'forecast.load_method','changed_rows':334,
        'original_signed_audit_code_and_numerical_forecasts_unchanged':True,
        'why':'Generic inherited CNN load_method residue; base_forecast, selected_model_id and numerical source equality already correctly identify HGB.'}
    (OUT/'metadata_erratum.json').write_text(json.dumps(erratum,ensure_ascii=False,indent=2)+'\n')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'three_layer_independent_verification.json').write_text(json.dumps(verifications,ensure_ascii=False,indent=2)+'\n')
    (OUT/'comparison_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('controlled_delta_linked_price_minus_previous_HGB',
        'actual_fee_reduction_vs_previous_HGB_pct','both_actual_fee_and_reversals_reduced')},indent=2))


if __name__=='__main__':
    main()
