"""Prepare the existing result2 template writer's input from the approved LP.

Use the bundled Python runtime for this export-only data preparation.
The original experiment and template-authoring code are imported unchanged.
"""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'data/results/exp005/soft-penalty-beta-0.1'
OUT = ROOT / 'reports/experiments/exp005'
WORK = ROOT / '.work/exp005'


def clock(slot):
    return f'{slot // 6:02}:{slot % 6 * 10:02}'


def periods(energy):
    edge = np.diff(np.r_[False, energy > 1e-6, False].astype(int))
    return [(int(a), int(b), float(energy[a:b].sum()))
            for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1), strict=True)]


def main():
    status = json.loads((RUN / 'status.json').read_text())
    if status['status'] != 'complete' or status['completed_days'] != 334:
        raise RuntimeError('A complete 334-day main policy is required before result2.xlsx export')
    import pandas as pd
    audit = pd.read_csv(OUT / 'soft_penalty/evidence/audit.csv')
    audit = audit[audit.beta == .1]
    assert len(audit) == 11 and audit.passed.all()
    run_audit = pd.read_csv(OUT / 'soft_penalty/evidence/runs.csv')
    assert int(run_audit[run_audit.beta == .1].iloc[0].completed_days) == 334
    z = dict(np.load(RUN / 'dispatch.npz'))
    price = np.load(OUT / 'strict_rerun/evidence/exp004_dispatch_2.npz')['price']
    assert z['grid'].shape == (334, 144) and z['states'].shape == (334, 145)
    dates = [(date(2025, 2, 1) + timedelta(days=i)).isoformat() for i in range(334)]
    fees = np.stack([z['grid'] * price, 5 * z['emergency'] * price], axis=-1)
    np.testing.assert_allclose(fees, z['fees'], atol=1e-6, rtol=0)
    battery, emergency, blocks, emergency_rows = [], [], [], []
    body = ['# 问题二指定四日结果', '主方案β=0.1，连续LP加周转惩罚。电量、SOC单位为kWh；费用为元。普通购电计划在每天零点锁定。']
    specified = {'2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21'}
    for i, day in enumerate(dates):
        daily_blocks = []
        for block in range(6):
            a, b = block * 24, (block + 1) * 24
            c, d = float(z['charge'][i, a:b].sum()), float(z['discharge'][i, a:b].sum())
            label = f'{clock(a)}-{clock(b)}'
            battery.append([day if block == 0 else None, label, c, d,
                            '0:00' if block == 0 else '24:00' if block == 1 else None,
                            float(z['states'][i, 0]) if block == 0 else float(z['states'][i, -1]) if block == 1 else None])
            daily_blocks.append([label, c, d])
            blocks.append({'date': day, 'interval': label, 'charge_kwh': c, 'discharge_kwh': d})
        events = periods(z['emergency'][i])
        np.testing.assert_allclose(sum(v for _, _, v in events), z['emergency'][i].sum(), atol=1e-5, rtol=0)
        for j, (a, b, value) in enumerate(events or [(0, 0, 0.)]):
            label = f'{clock(a)}-{clock(b)}' if b else '无'
            emergency.append([day if j == 0 else None, label, value])
            emergency_rows.append({'date': day, 'interval': label, 'emergency_kwh': value})
        if day in specified:
            body.extend([f'\n## {day}', '\n### 表1：计划购电量', '| 时间段 | 购电量 / kWh |', '|---|---:|'])
            for hour in [10, 12, 14, 16, 18, 20]:
                body.append(f'| {hour:02}:00-{hour:02}:10 | {z["grid"][i, hour * 6]:.4f} |')
            body.append(f'\n全天计划电量{z["grid"][i].sum():.4f} kWh，实际全部购电费{fees[i].sum():.4f}元。')
            body.extend(['\n### 表2：实际储能充放电', '| 时间段 | 充电 / kWh | 放电 / kWh |', '|---|---:|---:|'])
            body.extend(f'| {label} | {c:.4f} | {d:.4f} |' for label, c, d in daily_blocks)
            body.append(f'\n0:00储电量{z["states"][i, 0]:.4f} kWh，24:00储电量{z["states"][i, -1]:.4f} kWh。')
            body.extend(['\n### 表3：实际紧急购电', '| 时间段 | 紧急购电 / kWh |', '|---|---:|'])
            body.extend([f'| {clock(a)}-{clock(b)} | {v:.4f} |' for a, b, v in events] or ['| 无 | 0.0000 |'])
    payload = {'dates': dates, 'original': z['grid'].tolist(), 'final': z['grid'].tolist(),
               'fees': fees.sum(axis=(1, 2)).tolist(), 'battery': battery, 'emergency': emergency}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'workbook-2.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    (OUT / 'specified_dates.md').write_text('\n\n'.join(body) + '\n')
    for name, value in [('battery_blocks', blocks), ('emergency_periods', emergency_rows)]:
        (OUT / 'soft_penalty/evidence' / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    verification = {'result2.xlsx': {'days': 334, 'intervals_per_day': 144,
                    'total_cost': float(fees.sum()), 'violations': 0,
                    'strict_mutual_exclusion': False, 'beta': .1, 'model_class': 'LP',
                    'observed_overlap_intervals': int((np.minimum(z['charge'], z['discharge']) > 1e-6).sum()),
                    'max_raw_overlap_kwh': float(np.minimum(z['charge'], z['discharge']).max()),
                    'original_template_unchanged': True,
                    'template_sha256': hashlib.sha256((ROOT / 'data/templates/result2.xlsx').read_bytes()).hexdigest()}}
    (ROOT / 'data/results/exp005/verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps({'prepared_days': 334, 'battery_rows': len(battery), 'emergency_rows': len(emergency),
                      'total_cost': float(fees.sum())}))


if __name__ == '__main__':
    main()
