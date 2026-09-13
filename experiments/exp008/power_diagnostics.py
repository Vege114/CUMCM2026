"""Prepare exact power-series evidence required by the latest main template.

This exports data, not a final experiment report. It never smooths or samples
the annual trace. Random days only define the separate daily comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import verify_npz


def prepare(archive, output, *, scenario='2', initial_soc=1421.7991105135516,
            initial_power_kw=172.75999999999976, seed=20260912,
            role='development candidate; target has not been achieved'):
    path=Path(archive)
    out=Path(output)
    with np.load(path) as z:
        a={key:z[key].copy() for key in z.files}
    days=len(a['charge'])
    check=verify_npz(path,scenario,expected_days=days,initial_soc=initial_soc,
                     initial_power_kw=initial_power_kw)
    if not check['passed']:
        raise ValueError(f'Cannot export unverified power: {check["errors"]}')
    if days<4:
        raise ValueError('At least four complete days are required')
    if out.exists() and (out/'manifest.json').exists():
        raise FileExistsError('Keep prior evidence immutable; choose a new directory')
    out.mkdir(parents=True,exist_ok=True)
    # Interval endpoints match the original problem CSV convention.
    endpoints=pd.date_range('2025-02-01 00:10',periods=days*144,freq='10min')
    charge=6*a['charge'].ravel()
    discharge=6*a['discharge'].ravel()
    power=charge-discharge
    delta=np.diff(power)
    frame=pd.DataFrame(dict(interval_start=endpoints-pd.Timedelta(minutes=10),
        interval_end=endpoints,charge_power_kw=charge,discharge_power_kw=discharge,
        net_battery_power_kw=power,
        # The first formal point has no formal predecessor. Its warmup
        # boundary difference is kept separately in metrics, never filled 0.
        delta_power_kw=np.r_[np.nan,delta],
        soc_start_kwh=a['states'][:,:-1].ravel(),soc_end_kwh=a['states'][:,1:].ravel(),
        load_kw=a['actual'][...,0].ravel(),pv_kw=a['actual'][...,1].ravel()))
    frame.to_csv(out/'power_all_intervals.csv',index=False)
    chosen=np.sort(np.random.default_rng(seed).choice(days,size=4,replace=False))
    selected_dates=[]
    for day in chosen:
        date=(pd.Timestamp('2025-02-01')+pd.Timedelta(days=int(day))).strftime('%Y-%m-%d')
        selected_dates.append(date)
        frame.iloc[day*144:(day+1)*144].to_csv(out/f'power_{date}.csv',index=False)
    tol=1e-6
    absdelta=np.abs(delta)
    metrics=dict(days=days,points=len(power),delta_points=len(delta),
        mean_absolute_delta_kw=float(absdelta.mean()),rms_delta_kw=float(np.sqrt(np.mean(delta**2))),
        p95_absolute_delta_kw=float(np.quantile(absdelta,.95)),
        max_absolute_delta_kw=float(absdelta.max()),total_variation_kw=float(absdelta.sum()),
        direct_reversals=int(np.sum(((power[:-1]>tol)&(power[1:]<-tol))|
                                    ((power[:-1]<-tol)&(power[1:]>tol)))),
        power_limit_share=float(np.mean(np.abs(power)>=5000-tol)),
        jump_threshold_kw=1000.,large_jump_share=float(np.mean(absdelta>1000+tol)),
        warmup_boundary_delta_kw=float(power[0]-initial_power_kw),
        warmup_boundary_in_formal_delta_statistics=False,zero_tolerance_kw=tol,
        direction_reversals_nonidle=check['battery_metrics']['direction_reversals'],
        fees=check['billing'])
    manifest=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        role=role,scenario=scenario,start='2025-02-01',days=days,
        complete_365_day_trace=False,warmup_included=False,
        warmup_boundary_source_verified=False,
        warmup_boundary_reference='caller-supplied preceding power; check against its warmup archive before final reporting',
        annual_trace_label=f'{days}日正式评价；全部{days*144}个原始十分钟点',
        interval_convention='right endpoint',power_definition='6*(AC charge kWh-AC discharge kWh)',
        smoothing=False,annual_downsampling=False,
        random_day_seed=seed,random_day_algorithm='numpy.random.default_rng(seed).choice(days,4,replace=False); sort',
        random_dates=selected_dates,verification=check,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'power_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--scenario',default='2')
    p.add_argument('--initial-soc',type=float,default=1421.7991105135516)
    p.add_argument('--initial-power-kw',type=float,default=172.75999999999976)
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--role',default='development candidate; target has not been achieved')
    print(json.dumps(prepare(**vars(p.parse_args())),ensure_ascii=False,indent=2))
