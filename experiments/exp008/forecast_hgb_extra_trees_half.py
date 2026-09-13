"""One fixed 50/50 raw HGB/ExtraTrees blend, then shared causal postprocessing.

Both raw channels and all dates use the same predeclared weight. This candidate
was designed after development results were examined; it is not an untouched
test. No model fitting, blend-weight selection, or daily channel selection.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_hgb import (
    OUT as HGB, AbsoluteHGBStore, ArrayStore, array_hash, digest, save,
)
from experiments.exp008.forecast_absolute_extra_trees import (
    OUT as EXTRA, AbsoluteExtraTreesStore,
)
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, CONFIG
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.audited_store_bridge import run as bridge
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT / 'data/results/exp008/forecast_hgb_extra_trees_half'
COMPOSITE = ROOT / 'data/results/exp008/forecast_channel_composite'
RAW = 'hgb_extra_trees_half_raw'
RIDGE = 'hgb_extra_trees_half_ridge28'
PRIMARY = 'hgb_extra_trees_half_ridge28_memory'
WEIGHT = .5
GATE_KEYS = ('rmse_kw', 'high_price_rmse_kw', 'daily_energy_rmse_kwh',
             'cumulative_error_rmse_kwh')


class HGBExtraTreesHalfStore(ArrayStore):
    def __init__(self, stage=PRIMARY):
        meta = json.loads((OUT / 'provenance.json').read_text())
        path = OUT / f'{stage}.npz'
        if not meta['complete'] or digest(path) != meta['archives'][stage]:
            raise ValueError('The blend archive is incomplete or changed')
        with np.load(path) as z:
            super().__init__(z['values'], z['origins'], stage)


def prepare_protocol():
    if OUT.exists():
        raise FileExistsError('Existing evidence is immutable')
    inputs = [directory / name for directory in (HGB, EXTRA)
              for name in ('protocol.json', 'provenance.json', 'training_audit.json',
                           'causality_verification.json', 'independent_verification.json')]
    inputs += [HGB / 'direct_hgb_raw.npz', HGB / 'direct_hgb_ridge28_memory_half.npz',
               EXTRA / 'absolute_extra_trees_raw.npz',
               EXTRA / 'absolute_extra_trees_ridge28_memory_half.npz',
               EXTRA / 'resume_verification.json', COMPOSITE / 'memory.npz',
               COMPOSITE / 'summary.json']
    sources = [Path(__file__).resolve(),
               ROOT / 'experiments/exp008/forecast_absolute_hgb.py',
               ROOT / 'experiments/exp008/forecast_absolute_extra_trees.py',
               ROOT / 'experiments/exp008/forecast_calibration.py',
               ROOT / 'experiments/exp008/load_energy_memory.py',
               ROOT / 'experiments/exp008/forecast_shape_diagnostic.py',
               ROOT / 'experiments/exp008/audited_store_bridge.py']
    protocol = {
        'candidate_count': 1, 'model_id': PRIMARY, 'raw_weight_HGB': WEIGHT,
        'raw_weight_ExtraTrees': 1-WEIGHT, 'both_channels_same_weight_all334days': True,
        'raw_blend': '.5 * signed original absolute HGB raw + .5 * signed ExtraTrees raw',
        'postprocessing': 'same joint-channel Ridge28 fitted on this blended raw archive, then same nonrecursive .5 load memory',
        'no_postprocessed_output_blending': True,
        'no_weight_tuning_or_model_refitting_or_daywise_selection': True,
        'origin_range': '2025-02-01 through 2025-12-31, 334 complete days',
        'design_context': 'fixed blend proposed after observing development component metrics',
        'development_not_independent_test': True,
        'raw_model_causality': 'references signed independent audits for 22 original HGB and 22 ExtraTrees monthly models; no retraining here',
        'postprocessing_mutation_days': [31,32,59,90,151,243,364],
        'continuation_gate': {'keys': GATE_KEYS, 'comparator': 'original absolute HGB full pipeline',
                              'rule': 'all four metrics strictly improve and verification passes'},
        'conditional_bridge': 'one full334 unchanged tree28/q.8/state_buffer500 using own issued errors',
        'full_physical_planning': False, 'final_model_selection': False,
        'input_sha256': {str(p): digest(p) for p in inputs},
        'source_sha256': {str(p): digest(p) for p in sources},
        'source_data_sha256': Data().hashes,
    }
    OUT.mkdir(parents=True)
    save(OUT / 'protocol.json', protocol)
    for p in sources:
        target = OUT / 'source_archive' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    return protocol


def source_checks(protocol):
    for key in ('source_sha256', 'input_sha256'):
        assert all(digest(path) == sha for path, sha in protocol[key].items())
    for directory in (HGB, EXTRA):
        assert json.loads((directory / 'independent_verification.json').read_text())['passed']
        assert json.loads((directory / 'causality_verification.json').read_text())['passed']


def main():
    protocol = prepare_protocol()
    source_checks(protocol)
    hgb = AbsoluteHGBStore('direct_hgb_raw')
    extra = AbsoluteExtraTreesStore('absolute_extra_trees_raw')
    np.testing.assert_array_equal(hgb.origins, extra.origins)
    np.testing.assert_array_equal(hgb.origins, np.arange(31,365)*144)
    values = WEIGHT*hgb.values+(1-WEIGHT)*extra.values
    raw = ArrayStore(values, hgb.origins.copy(), RAW)
    data = Data()
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False,
                            _base_store=raw, directory=OUT/'unused_cache')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    ridge.name, memory.name = RIDGE, PRIMARY
    for row in ridge.audit:
        row['actual_base_model_id'] = RAW
    for row in memory.audit:
        row['underlying_forecast'] = 'fixed_raw_HGB_ExtraTrees_half_then_Ridge28'
    truth = data.actual[raw.origins[:,None]+np.arange(144), :2]
    archives = {}
    for store in (raw, ridge, memory):
        path = OUT / f'{store.name}.npz'
        np.savez_compressed(path, values=store.values, origins=store.origins,
                            errors_kw=truth-store.values)
        archives[store.name] = digest(path)
    save(OUT/'ridge_audit.json', ridge.audit)
    save(OUT/'memory_audit.json', memory.audit)
    checks = []
    for day in protocol['postprocessing_mutation_days']:
        i = day-31
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day*144:] += [70000.,30000.]
        future_hgb, future_extra = hgb.values.copy(), extra.values.copy()
        future_hgb[i+1:] += [80000.,20000.]
        future_extra[i+1:] += [30000.,90000.]
        future_raw = WEIGHT*future_hgb+(1-WEIGHT)*future_extra
        np.testing.assert_array_equal(future_raw[:i+1], raw.values[:i+1])
        a,_ = _calibrate(data.actual, raw.origins, raw.values, i, CONFIG['ridge_28'])
        b,_ = _calibrate(changed.actual, raw.origins, future_raw, i, CONFIG['ridge_28'])
        np.testing.assert_array_equal(a, b)
        future_ridge = ridge.values.copy()
        future_ridge[i+1:] += [40000.,90000.]
        a,_ = correct_day(data.actual, raw.origins, ridge.values, i)
        b,_ = correct_day(changed.actual, raw.origins, future_ridge, i)
        np.testing.assert_array_equal(a, b)
        checks.append({'day':day, 'asymmetric_future_actual_and_both_source_forecasts':True,
                       'raw_prefix_ridge_and_memory_exactly_unchanged':True})
    assert all(row['history_last_label'] is None or row['history_last_label'] < row['origin']
               for row in ridge.audit)
    assert all(row['prior_label_end_exclusive'] is None or row['prior_label_end_exclusive'] <= row['origin']
               for row in memory.audit)
    save(OUT/'causality_verification.json', {'passed':True, 'checks':checks,
          'all334_postprocessing_labels_before_issue':True,
          'raw_training_causality_inherited_from_signed_model_audits':True})
    with np.load(COMPOSITE/'memory.npz') as z:
        np.testing.assert_array_equal(z['origins'], raw.origins)
        composite = z['values'].copy()
    metrics = []
    for name,v in [('original_HGB_full', AbsoluteHGBStore().values),
                   ('ExtraTrees_full', AbsoluteExtraTreesStore().values),
                   ('best_fixed_channel_composite_full', composite),
                   (RAW, raw.values), (RIDGE, ridge.values), (PRIMARY, memory.values)]:
        for channel,e in [('load',v[:,:,0]-truth[:,:,0]), ('pv',v[:,:,1]-truth[:,:,1]),
                          ('net',(v[:,:,0]-v[:,:,1])-(truth[:,:,0]-truth[:,:,1]))]:
            metrics.append({'name':name, 'channel':channel, **score(e,data.fixed_price)})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False)
    net = {r['name']:r for r in metrics if r['channel']=='net'}
    gate_checks = {k:net[PRIMARY][k] < net['original_HGB_full'][k] for k in GATE_KEYS}
    gate = all(gate_checks.values())
    source_checks(protocol)
    save(OUT/'provenance.json', {'complete':True, 'model_id':PRIMARY, 'archives':archives,
          'raw_HGB_values_sha256':array_hash(hgb.values),
          'raw_ExtraTrees_values_sha256':array_hash(extra.values),
          'raw_exact_half_blend':True, 'protocol_sha256':digest(OUT/'protocol.json'),
          'source_and_input_hashes_unchanged':True})
    save(OUT/'summary.json', {'complete':True, 'metrics':metrics,
          'predictive_gate_passed':gate, 'gate_checks':gate_checks,
          'causality_verified':True, 'source_hashes_verified':True,
          'development_not_independent_test':True, 'final_model_selection':False})
    print(json.dumps({'net_metrics':net, 'gate':gate}, indent=2), flush=True)
    if gate:
        bridge(memory, OUT/'lp_bridge', [OUT/'summary.json', OUT/'protocol.json',
                                        OUT/f'{PRIMARY}.npz', OUT/'causality_verification.json',
                                        HGB/'provenance.json', EXTRA/'provenance.json'])
    source_checks(protocol)


if __name__ == '__main__':
    main()
