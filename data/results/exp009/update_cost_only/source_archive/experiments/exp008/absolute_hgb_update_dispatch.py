"""Full-year fixed Q3/Q4-3 linkage of the new absolute HGB forecast pipeline.

Only the midnight load/PV model changes. The legal releases, same-issue
historical paths, adjustment proxy, actual settlement, and executor are fixed.
"""
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.forecast_absolute_hgb import OUT as FORECAST_OUT, PRIMARY
from experiments.exp008.forecast_override_adapter import AuditedForecastOverride
from experiments.exp008.issued_residual_paths import issued_error_paths
from experiments.exp008.planner import Settings
from experiments.exp008.run import OUT as RESULTS, initial_state, run_case
from experiments.exp008.verify import verify_npz

OUT = RESULTS/'absolute_hgb_update_dispatch'
BASELINE = RESULTS/'update_value_diagnostic/future1.5_d334_issued'
SETTINGS = Settings(future_shortfall_weight=1.5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


class LinkedHGBForecasts(AbsoluteHGBForecasts):
    def get(self, day, slot=0, scenario='2'):
        r = super().get(day, slot, scenario)
        if day >= 31:
            r['audit']['load_method'] = 'direct_absolute_HGB_Ridge28_memory_half_remaining_trajectory_with_existing_causal_intraday_bias'
        return r


def override(data=None):
    return AuditedForecastOverride(LinkedHGBForecasts(data=data), PRIMARY,
        [Path(__file__), FORECAST_OUT/'provenance.json', FORECAST_OUT/f'{PRIMARY}.npz',
         FORECAST_OUT/'training_audit.json', FORECAST_OUT/'ridge28_audit.json',
         FORECAST_OUT/'memory_half_audit.json'])


def causal_checks():
    rows = []
    for scenario in ('3', '4-3'):
        for day, slot in ((31, 0), (32, 36), (60, 72), (180, 108)):
            before = override()
            changed = copy.copy(before.data)
            cutoff = day*144+slot
            changed.actual = before.data.actual.copy()
            changed.actual[cutoff:] += [100000., 50000., 1000.]
            changed._forecasts = {k: np.asarray(v).copy()+(90000. if k>cutoff else 0.)
                                  for k, v in before.data.forecasts.items()}
            after = override(changed)
            a, b = before.get(day, slot, scenario), after.get(day, slot, scenario)
            ra = issued_error_paths(before, day, slot, scenario=scenario)
            rb = issued_error_paths(after, day, slot, scenario=scenario)
            for key in ('load_kw', 'pv_kw', 'price'):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ('errors_kw', 'errors_kwh', 'price_errors', 'origins', 'label_stops_exclusive'):
                np.testing.assert_array_equal(ra[key], rb[key])
            assert not b['audit']['known_future_price']
            assert ra['label_stops_exclusive'].max() <= day*144
            rows.append({'scenario': scenario, 'day': day, 'slot': slot,
                'future_actual_and_unreleased_PV_mutation_identical': True,
                'same_issue_history_identical': True})
    result = {'passed': True, 'checks': rows,
              'raw_HGB_and_Ridge_memory_training_causality': str(FORECAST_OUT/'causality_verification.json')}
    save(OUT/'causality_verification.json', result)


def compare(scenario):
    run_case('absolute_hgb_update_dispatch/full334', scenario, method='joint',
             settings=SETTINGS, days=334, deadband=20., updates=True,
             issued_residuals=True, forecast_override=override())
    directory = OUT/'full334'/scenario
    soc, power = initial_state(scenario)
    args = dict(scenario=scenario, expected_days=334, initial_soc=soc,
                initial_power_kw=power, initial_mode=int(np.sign(power)))
    new = verify_npz(directory/f'dispatch_{scenario}.npz',
                     audit_path=directory/'audit.json', **args)
    old = verify_npz(BASELINE/scenario/f'dispatch_{scenario}.npz',
                     audit_path=BASELINE/scenario/'audit.json', **args)
    assert new['passed'] and old['passed']
    audit = json.loads((directory/'audit.json').read_text())
    for row in audit:
        assert row['residual_paths']['model_id'] == PRIMARY
        assert row['residual_paths']['historical_predictions_use_same_override']
        assert row['residual_paths']['label_stops_exclusive'][-1] <= row['day']*144
        assert not row['forecast']['known_future_price']
    changes = {key:new['billing'][key]-old['billing'][key] for key in old['billing']}
    result = {'scenario':scenario,'days':334,'baseline':old,'candidate':new,
        'cost_change_yuan':new['recomputed_total_cost']-old['recomputed_total_cost'],
        'cost_change_pct':100*(new['recomputed_total_cost']/old['recomputed_total_cost']-1),
        'billing_component_changes':changes,
        'reversals_change':new['battery_metrics']['direction_reversals']-old['battery_metrics']['direction_reversals'],
        'own_continuous_soc':True,'all_issue_model_ids_and_label_stops_verified':True,
        'Q2_goal_applied':False,'final_model_selection':False}
    save(directory/'paired_findings.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','candidate')},ensure_ascii=False),flush=True)
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Existing evidence is immutable')
    OUT.mkdir(parents=True)
    sources = ['absolute_hgb_update_dispatch','absolute_hgb_forecast_adapter',
               'forecast_absolute_hgb','forecast_override_adapter','issued_residual_paths',
               'forecast','planner','run','verify']
    hashes={}
    for name in sources:
        path=Path(__file__).parent/f'{name}.py'
        shutil.copy2(path,OUT/path.name)
        hashes[name]=digest(path)
    save(OUT/'protocol.json', {'scenarios':['3','4-3'],'days':334,
        'predeclared_full_year_no_February_performance_gate':True,
        'initial_gate':'causal issue/history perturbation checks only',
        'model_id':PRIMARY,'settings':asdict(SETTINGS),'deadband_kwh':20.,
        'same_issue_residuals':True,'updates':[0,36,72,108],
        'baseline':str(BASELINE),'source_hashes':hashes,
        'forecast_identity':override().forecast_identity(),
        'development_year_not_independent_test':True,
        'final_report_generated':False,'Q2_goal_applied':False})
    causal_checks()
    annual=[compare(scenario) for scenario in ('3','4-3')]
    save(OUT/'summary.json',{'complete':True,'annual':annual,'final_model_selection':False})


if __name__=='__main__':
    main()
