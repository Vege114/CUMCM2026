"""Causal shared forecast/planning/replay pipeline for Q2, Q3, Q4-2 and Q4-3."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.common.neural_v2.physics import deterministic, execute as old_execute, settle
from .issued_residual_paths import issued_error_paths
from .planner import Settings, execute, plan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008'


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def forecast_identity(forecasts):
    """Validate a store-backed override's declared identity before cache use.

    Prediction values and origins are checked independently, rather than
    trusting a model name that could refer to different cached forecasts.
    The override keeps the same get/_observed/net_error_paths/data protocol.
    """
    for name in ('get', '_observed', 'net_error_paths', 'forecast_identity'):
        if not callable(getattr(forecasts, name, None)):
            raise TypeError(f'forecast_override must provide callable {name}')
    identity = dict(forecasts.forecast_identity())
    if not isinstance(identity.get('model_id'), str) or not identity['model_id'].strip():
        raise ValueError('forecast_override requires an explicit nonempty model_id')
    for key, array in (('prediction_values_sha256', forecasts.store.values),
                       ('prediction_origins_sha256', forecasts.store.origins)):
        if identity.get(key) != _array_hash(array):
            raise ValueError(f'forecast_override {key} does not match the actual forecast store')
    # Reject non-JSON metadata before any result directory or archive is written.
    json.dumps(identity, sort_keys=True, allow_nan=False)
    return identity


def initial_state(scenario):
    with np.load(ROOT / f'data/results/exp002/warmup_{scenario}.npz') as z:
        return float(z['states'][-1,-1]), float(6*(z['charge'][-1,-1]-z['discharge'][-1,-1]))


def run_case(case, scenario, *, method='joint', settings=Settings(), deadband=20.,
             ramp=None, seed=42, updates=True, days=334, charge_mask=None,
             calibration=None, quantile=.8, issued_residuals=False,
             forecast_override=None):
    from .unified_forecast import UnifiedForecasts
    if forecast_override is None:
        forecasts = UnifiedForecasts(seed=seed,calibration=calibration)
        override_identity = None
    else:
        forecasts = forecast_override
        override_identity = forecast_identity(forecasts)
        if int(forecasts.seed) != seed:
            raise ValueError('forecast_override seed differs from the explicit run seed')
        if calibration is not None and calibration != getattr(forecasts, 'calibration', None):
            raise ValueError('calibration argument conflicts with forecast_override')
    data = forecasts.data
    directory = OUT / case / scenario
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(case=case, scenario=scenario, method=method, settings=asdict(settings),
                  deadband=deadband, ramp=ramp, seed=seed, updates=updates, days=days,
                  calibration=calibration,quantile=quantile,
                  issued_residuals=issued_residuals,
                  charge_mask=None if charge_mask is None else np.asarray(charge_mask).tolist())
    if override_identity is not None:
        config['forecast_override'] = override_identity
        config['effective_forecast_calibration'] = getattr(forecasts, 'calibration', None)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob('*.py')}
    signature = hashlib.sha256(json.dumps([config, source_hashes], sort_keys=True).encode()).hexdigest()
    existing = directory / 'completion.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        if old['signature'] == signature:
            print('CACHED', case, scenario, flush=True)
            return old
        raise RuntimeError(f'Refusing to overwrite changed experiment {directory}; use a new case name')
    began = perf_counter()
    soc, previous_power = initial_state(scenario)
    previous_mode = int(np.sign(previous_power))
    parts, rows, audits = [], [], []
    for day in range(31, 31+days):
        day_start = perf_counter()
        original = None
        final = np.zeros(144)
        states = np.empty(145)
        states[0] = soc
        components = {key: np.zeros(144) for key in ('charge','discharge','emergency','surplus')}
        boundaries = [0,36,72,108,144] if updates and scenario in ('3','4-3') else [0,144]
        day_audits = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            forecast = forecasts.get(day, slot=start, scenario=scenario)
            original_remainder = None if start == 0 else original[start:]
            errors = None
            if method == 'old_dispatch':
                purchases, meta = deterministic(forecast['load_kw'], forecast['pv_kw'],
                                                forecast['price'], states[start], original_remainder)
            else:
                errors = (issued_error_paths(forecasts, day, start, scenario=scenario)
                          if issued_residuals and scenario in ('3', '4-3')
                          else forecasts.net_error_paths(day, scenario=scenario))
                paths = errors['errors_kwh'] if isinstance(errors,dict) else errors
                if method=='quantile':
                    paths=np.quantile(paths,quantile,axis=0,keepdims=True) if len(paths) else np.zeros((1,144))
                planned = plan(forecast, states[start], paths, original_remainder,
                               settings=settings, final_day=day==364,
                               next_update_slot=stop-start if updates and scenario in ('3', '4-3') else None,
                               charge_mask=None if charge_mask is None else charge_mask[start:])
                purchases, meta = planned['purchase'], planned['metadata']
            if start == 0:
                original = purchases.copy()
            final[start:] = purchases
            # Actual values are consumed only after this information-time plan exists.
            observed = data.actual[day*144+start:day*144+stop]
            trading_price = observed[:,2] if scenario.startswith('4') else data.fixed_price[start:stop]
            if method == 'old_dispatch':
                c,d,e,w,s = old_execute(final[start:stop],observed[:,0],observed[:,1],states[start])
                executed = dict(charge=c,discharge=d,emergency=e,surplus=w,states=s)
            else:
                executed = execute(final[start:stop],observed,trading_price,states[start],
                                   charge_deadband=deadband,ramp_kw=ramp,
                                   previous_power=previous_power,previous_mode=previous_mode,
                                   charge_mask=None if charge_mask is None else charge_mask[start:stop])
            for key in components:
                components[key][start:stop] = executed[key]
            states[start:stop+1] = executed['states']
            previous_power = float(6*(executed['charge'][-1]-executed['discharge'][-1]))
            signs = np.sign(executed['charge']-executed['discharge'])
            if np.any(signs):
                previous_mode = int(signs[signs != 0][-1])
            row_audit = dict(day=day,slot=start,information_cutoff=day*144+start,
                                   forecast=forecast['audit'],solver=meta,
                                   original_locked=start==0,executed_until=day*144+stop)
            if isinstance(errors, dict):
                row_audit.update(residual_paths=errors['audit'],
                                 training_origins=np.asarray(errors['origins']).tolist(),
                                 max_observed_index=errors['audit']['max_observed_index'])
            if override_identity is not None:
                row_audit['forecast_override'] = override_identity
                row_audit['issued_prediction_sha256'] = {
                    key: _array_hash(forecast[key]) for key in ('load_kw', 'pv_kw', 'price')}
                if isinstance(errors, dict):
                    row_audit['residual_paths'] = {**errors['audit'],
                        'source': 'same_selected_override_adapter_with_labelled_January_periodic_cold_start',
                        'model_id': override_identity['model_id'],
                        'prediction_values_sha256': override_identity['prediction_values_sha256'],
                        'errors_kwh_sha256': _array_hash(errors['errors_kwh']),
                        'price_errors_sha256': _array_hash(errors['price_errors']),
                        'historical_predictions_use_same_override': True}
            day_audits.append(row_audit)
        actual = data.actual[day*144:(day+1)*144].copy()
        price = actual[:,2].copy() if scenario.startswith('4') else data.fixed_price.copy()
        fees = settle(original,final,components['emergency'],price)
        detail = dict(original=original,final=final,**components,states=states,fees=fees,
                      actual=actual,price=price)
        soc = float(states[-1])
        parts.append(detail)
        rows.append(dict(day=day,date=str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
                         scenario=scenario,planned_cost=float(fees[:,0].sum()),
                         up_cost=float(fees[:,1].sum()),down_cost=float(fees[:,2].sum()),
                         emergency_cost=float(fees[:,3].sum()),total_cost=float(fees.sum()),
                         emergency_kwh=float(components['emergency'].sum()),
                         throughput_kwh=float((components['charge']+components['discharge']).sum()),
                         initial_soc=float(states[0]),final_soc=soc,seconds=perf_counter()-day_start))
        audits.extend(day_audits)
        if (day-30) % 30 == 0:
            print(case,scenario,day-30,'days',round(sum(r['total_cost'] for r in rows),2),flush=True)
    if override_identity is not None and forecast_identity(forecasts) != override_identity:
        raise RuntimeError('forecast_override identity changed during execution; refusing to archive mixed models')
    archive = {key:np.stack([part[key] for part in parts]) for key in parts[0]}
    archive['days'] = np.arange(31,31+days)
    np.savez_compressed(directory / f'dispatch_{scenario}.npz', **archive)
    pd.DataFrame(rows).to_csv(directory/'daily.csv', index=False)
    (directory/'audit.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2))
    completion = dict(signature=signature,config=config,source_hashes=source_hashes,
                      wall_seconds=perf_counter()-began,complete=days==334,
                      total_cost=sum(row['total_cost'] for row in rows))
    existing.write_text(json.dumps(completion,ensure_ascii=False,indent=2))
    print('DONE',case,scenario,completion['total_cost'],flush=True)
    return completion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--scenarios',nargs='+',default=['3','4-2','4-3'])
    parser.add_argument('--method',choices=['old_dispatch','joint','quantile'],default='joint')
    parser.add_argument('--days',type=int,default=334)
    parser.add_argument('--deadband',type=float,default=20.)
    parser.add_argument('--ramp',type=float)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--no-updates',action='store_true')
    parser.add_argument('--calibration')
    parser.add_argument('--quantile',type=float,default=.8)
    parser.add_argument('--fixed-blocks',action='store_true')
    parser.add_argument('--issued-residuals',action='store_true',
                        help='Use prior complete days at the same forecast release in Q3/Q4-3')
    parser.add_argument('--future-shortfall-weight',type=float,
                        help='Q3/Q4-3 planning-only shortfall weight after the next legal update (e.g. 1.5)')
    args=parser.parse_args()
    for scenario in args.scenarios:
        t=np.arange(144)
        mask=((t<34)|((t>=60)&(t<96))|(t>=131)) if args.fixed_blocks else None
        run_case(args.case,scenario,method=args.method,days=args.days,deadband=args.deadband,
                 ramp=args.ramp,seed=args.seed,updates=not args.no_updates,
                 calibration=args.calibration,quantile=args.quantile,charge_mask=mask,
                 issued_residuals=args.issued_residuals,
                 settings=Settings(future_shortfall_weight=args.future_shortfall_weight))


if __name__=='__main__':
    main()
