"""Independent reconstruction of the fixed raw blend and its billing bridge."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_hgb_extra_trees_half import (
    OUT, HGB, EXTRA, COMPOSITE, RAW, RIDGE, PRIMARY, GATE_KEYS,
    HGBExtraTreesHalfStore, source_checks,
)
from experiments.exp008.forecast_absolute_hgb import digest, save
from experiments.exp008.forecast_calibration import _calibrate, CONFIG
from experiments.exp008.verify import verify_npz
from experiments.problem2.exp003.data import Data


def audit():
    destination = OUT/'independent_verification.json'
    if destination.exists():
        raise FileExistsError('Preserve the completed independent audit')
    protocol = json.loads((OUT/'protocol.json').read_text())
    source_checks(protocol)
    data = Data()
    raw, ridge, memory = [HGBExtraTreesHalfStore(k) for k in (RAW,RIDGE,PRIMARY)]
    with np.load(HGB/'direct_hgb_raw.npz') as hgb, np.load(EXTRA/'absolute_extra_trees_raw.npz') as extra:
        np.testing.assert_array_equal(raw.origins, hgb['origins'])
        np.testing.assert_array_equal(raw.origins, extra['origins'])
        np.testing.assert_array_equal(raw.values, (hgb['values']+extra['values'])/2)
    np.testing.assert_array_equal(raw.origins, np.arange(31,365)*144)
    for i,origin in enumerate(raw.origins):
        expected,causal = _calibrate(data.actual,raw.origins,raw.values,i,CONFIG['ridge_28'])
        np.testing.assert_array_equal(expected,ridge.values[i])
        assert causal['history_last_label'] is None or causal['history_last_label'] < origin
        if i:
            last = int(raw.origins[i-1])
            delta = .5*np.mean(data.actual[last:last+144,0]-ridge.values[i-1,:,0])
            expected[:,0] = np.maximum(expected[:,0]+delta,0.)
        np.testing.assert_array_equal(expected,memory.values[i])
    truth = data.actual[raw.origins[:,None]+np.arange(144),:2]
    for store in (raw,ridge,memory):
        assert store.values.shape == (334,144,2)
        assert np.isfinite(store.values).all() and np.min(store.values) >= 0
        with np.load(OUT/f'{store.name}.npz') as z:
            np.testing.assert_array_equal(z['errors_kw'],truth-store.values)
    paths = {'original_HGB_full':HGB/'direct_hgb_ridge28_memory_half.npz',
             'ExtraTrees_full':EXTRA/'absolute_extra_trees_ridge28_memory_half.npz',
             'best_fixed_channel_composite_full':COMPOSITE/'memory.npz',
             PRIMARY:OUT/f'{PRIMARY}.npz'}
    recorded = json.loads((OUT/'summary.json').read_text())
    claimed = {r['name']:r for r in recorded['metrics'] if r['channel']=='net'}
    recomputed = {}
    dates = pd.date_range('2025-02-01','2025-12-31')
    monthly = []
    for name,path in paths.items():
        with np.load(path) as z:
            np.testing.assert_array_equal(z['origins'],raw.origins)
            error = (z['values']-truth)[:,:,0]-(z['values']-truth)[:,:,1]
        for month in [None,*range(2,13)]:
            e = error if month is None else error[dates.month==month]
            row = {'rmse_kw':float(np.sqrt(np.mean(e**2))),
                'high_price_rmse_kw':float(np.sqrt(np.mean(e[:,data.fixed_price>=np.quantile(data.fixed_price,.75)]**2))),
                'daily_energy_rmse_kwh':float(np.sqrt(np.mean((e.sum(axis=1)/6)**2))),
                'cumulative_error_rmse_kwh':float(np.sqrt(np.mean((e.cumsum(axis=1)/6)**2)))}
            if month is None:
                recomputed[name] = row
                for key,value in row.items():
                    np.testing.assert_allclose(value,claimed[name][key],atol=1e-8,rtol=0)
            else:
                monthly.append({'name':name,'month':month,**row})
    gate = all(recomputed[PRIMARY][k] < recomputed['original_HGB_full'][k] for k in GATE_KEYS)
    assert gate == recorded['predictive_gate_passed']
    causal = json.loads((OUT/'causality_verification.json').read_text())
    assert causal['passed'] and len(causal['checks']) == 7
    # These checks independently reconstruct every postprocessing date; the
    # signed raw model checks cover model fitting, not an assertion of a test set.
    result = {'passed':True,'auditor_source_sha256':digest(__file__),
              'all334_raw_origins_and_exact_half_blend_verified':True,
              'all334_Ridge28_and_independent_nonrecursive_half_memory_verified':True,
              'all334_issued_errors_verified':True,'all_four_gate_metrics_recomputed':recomputed,
              'raw22_models_per_family_causality_from_signed_source_audits':True,
              'seven_asymmetric_postprocessing_future_checks_present_and_passed':True,
              'signed_sources_and_inputs_unchanged':True,
              'predictive_gate_passed':gate,'development_not_independent_test':True}
    if gate:
        case = OUT/f'lp_bridge/{PRIMARY}_tree28_q08_buffer500_334days'
        candidate = verify_npz(case/'dispatch_2.npz',audit_path=case/'source_corrected_audit.json')
        assert candidate['passed']
        paired = json.loads((OUT/'lp_bridge/paired_findings.json').read_text())
        assert candidate['source_sha256'] == paired['candidate']['source_sha256']
        np.testing.assert_allclose(candidate['recomputed_total_cost'],paired['candidate']['recomputed_total_cost'],rtol=0,atol=1e-8)
        result['independent_dispatch_verification'] = candidate
        result['bridge_difference_vs_ExtraTrees_yuan'] = candidate['recomputed_total_cost']-13147867.580952927
        result['bridge_goal_passed'] = candidate['goal']['passed']
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    source_checks(protocol)
    save(destination,result)
    (OUT/'independent_auditor_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('independent_dispatch_verification','all_four_gate_metrics_recomputed')},indent=2))


if __name__ == '__main__':
    audit()
