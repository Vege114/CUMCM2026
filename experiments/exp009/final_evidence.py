"""Independent completed-archive audit and exact static battery figures for exp009.

Reads existing arrays only: no forecasting, fitting, optimization or dispatch.
Missing/incomplete cases are pending, never annualized. Run again when their
completion markers exist; --require-all refuses a partial final deliverable.
Only this experiment's evidence/battery and figures/battery are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.exp008.verify import verify_arrays

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/experiments/exp009/evidence/battery'
FIG = ROOT / 'reports/experiments/exp009/figures/battery'
PROTOCOL = ROOT / 'experiments/exp009/protocol.json'
OLD = ROOT / 'data/results/exp008'
NEW = ROOT / 'data/results/exp009'
Q2OLD = OLD / 'mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
Q4OLD = OLD / 'q4_hgb_linked_price_physical'
UPO = OLD / 'absolute_hgb_update_dispatch/full334'
UPN = NEW / 'update_cost_only/full334'
CASES = {
    '1': (OLD/'q1/revised.json', NEW/'q1/cost_only.json', None, None),
    '2': (Q2OLD/'dispatch.npz', NEW/'q2_cost_only/dispatch.npz',
          NEW/'q2_cost_only/summary.json', ROOT/'data/results/exp003/warmup_2.npz'),
    '3': (UPO/'3/dispatch_3.npz', UPN/'3/dispatch_3.npz',
          UPN/'3/completion.json', ROOT/'data/results/exp002/warmup_3.npz'),
    '4-2': (Q4OLD/'full334/dispatch_4-2.npz', NEW/'q4_2_cost_only/full334/dispatch_4-2.npz',
            NEW/'q4_2_cost_only/completion.json', ROOT/'data/results/exp002/warmup_4-2.npz'),
    '4-3': (UPO/'4-3/dispatch_4-3.npz', UPN/'4-3/dispatch_4-3.npz',
            UPN/'4-3/completion.json', ROOT/'data/results/exp002/warmup_4-3.npz'),
}
AUDITS = {'2': ('planning_audit.json', 'planning_audit.json'),
          '3': ('audit.json', 'audit.json'), '4-2': ('source_corrected_audit.json', 'audit.json'),
          '4-3': ('audit.json', 'audit.json')}
SEED = 20260912
DATES = pd.date_range('2025-02-01', '2025-12-31')
RANDOM = np.sort(np.random.default_rng(SEED).choice(334, 4, replace=False))
SPECIFIED = [pd.Timestamp(x) for x in ('2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21')]
SPECIFIED_IDX = np.array([DATES.get_loc(x) for x in SPECIFIED])
ETA, TOL = np.sqrt(.9), 1e-6
COLORS = {'exp008': '#7d8187', 'exp009': '#245a82'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ref(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': digest(path)}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def raw_sources():
    names = ['附件1.csv', '附件2_小区负载.csv', '附件2_光伏发电实际功率.csv', '附件3.csv', '附件4.csv']
    files = [ROOT/'data/raw'/name for name in names]
    original_validation = ROOT/'reports/experiments/exp008/evidence/data_validation.json'
    original_hashes = {x['source']['path']: x['source']['sha256'] for x in read(original_validation)['files']}
    for path in files:
        assert digest(path) == original_hashes[str(path.relative_to(ROOT))], 'raw CSV changed from exp008'
    a1, ld, pv, _, price = [pd.read_csv(p) for p in files]
    for frame in (ld, pv, price):
        assert frame.shape == (365, 145)
        np.testing.assert_array_equal(pd.to_datetime(frame.iloc[:, 0]).to_numpy(),
                                      pd.date_range('2025-01-01', periods=365).to_numpy())
    assert a1.shape == (144, 4)
    assert all(list(x.columns[1:]) == list(ld.columns[1:]) for x in (pv, price))
    def end_minutes(value):
        text = str(value)
        fields = text.replace('+1', '').split(':')
        return int(fields[0])*60+int(fields[1])+(1440 if '+1' in text else 0)
    # Raw source mixes HH:MM and HH:MM:SS and names day end0:00+1.
    np.testing.assert_array_equal([end_minutes(x) for x in a1.iloc[:, 0]], np.arange(1, 145)*10)
    q1actual = a1.iloc[:, 2:4].to_numpy(float)[None, :, :]
    actual = np.stack((ld.iloc[31:, 1:].to_numpy(float), pv.iloc[31:, 1:].to_numpy(float)), axis=-1)
    prices = {'fixed': a1.iloc[:, 1].to_numpy(float), 'realized': price.iloc[31:, 1:].to_numpy(float)}
    return actual, q1actual, prices, {'sources': [ref(p) for p in files], 'raw_days': 365,
        'all_five_raw_byte_hashes_equal_exp008': True, 'exp008_validation_source': ref(original_validation),
        'evaluation_days': 334, 'evaluation_start': '2025-02-01', 'evaluation_end': '2025-12-31',
        'raw_date_order_and_slot_headers_checked': True, 'slots_per_day': 144,
        'interval_minutes': 10, 'interval_convention': 'right endpoint; last slot ends at24:00'}


def q1_arrays(path, q1actual, price):
    result = read(path)
    q = {k: np.asarray(v, float) for k, v in result['trajectory'].items()}
    g = q['g'][None, :]
    a = {'original': g, 'final': g.copy(), 'charge': q['c'][None, :]/6,
         'discharge': q['d'][None, :]/6, 'emergency': np.zeros_like(g),
         'surplus': q['w'][None, :], 'states': q['E'][None, :], 'actual': q1actual,
         'price': price[None, :], 'fees': np.stack((g*price, np.zeros_like(g),
                    np.zeros_like(g), np.zeros_like(g)), axis=-1)}
    assert q['c'].shape == (144,) and q['E'].shape == (145,)
    on, off = [], []
    for key, power in (('zc', q['c']), ('zd', q['d'])):
        z = q[key]
        assert np.max(np.abs(z-np.rint(z))) < 1e-6
        assert np.all(power >= z-2e-5) and np.all(power <= 5000*z+2e-5)
        z = np.rint(z).astype(int)
        for value, target in ((1, on), (0, off)):
            starts = np.flatnonzero((z == value) & (np.roll(z, 1) != value))
            target.extend(next(j for j in range(1, 145) if z[(i+j)%144] != value) for i in starts)
            if len(starts) == 0 and np.all(z == value):
                target.append(144)
    power = q['c']-q['d']
    cyclic = np.abs(power-np.roll(power, 1))
    assert np.max(q['zc']+q['zd']) <= 1+TOL
    assert min(on, default=144) >= 3 and min(off, default=144) >= 2
    assert cyclic.max() <= 1000+2e-5
    assert result['config']['ramp'] == 1000 and result['config']['up'] == 3 and result['config']['down'] == 2
    assert result['config']['eta_rt'] == .9
    assert abs(float(a['fees'].sum())-result['stages'][-1]['cost']) < 1e-6
    return a, {'source': ref(path), 'stages': result['stages'], 'configuration': result['config'],
        'cyclic_total_variation_kw': float(cyclic.sum()), 'cyclic_mode_starts': len(on),
        'cyclic_min_on_minutes': 10*min(on, default=144), 'cyclic_min_off_minutes': 10*min(off, default=144),
        'max_cyclic_step_kw': float(cyclic.max()), 'retained_minimum_active_power_kw': 1,
        'mode_power_linking_and_engineering_constraints_passed': True,
        'unit_note': 'Q1 stored c/d already kW; converted once to kWh for shared verifier'}


def forecast_identity(scenario, before, after):
    """Compare frozen inputs or per-issue array hashes, not just model names."""
    rows, sources = [], []
    if scenario == '2':
        provenance = read(NEW/'q2_cost_only/provenance.json')
        for name, expected in provenance['input_sha256'].items():
            assert digest(ROOT/name) == expected, name
        assert provenance['protocol_sha256'] == digest(PROTOCOL)
        for name, expected in provenance['source_sha256'].items():
            assert digest(ROOT/name) == expected, name
        for i, (old, new) in enumerate(zip(before, after, strict=True)):
            day = 31+i
            assert old['day'] == new['day'] == day
            assert old['forecast'] == new['forecast'] and old['history'] == new['history']
            source = Q2OLD/f'planning_day{day}.npz'
            assert new['external_source_sha256'] == digest(source)
            src, own = load(source), load(NEW/f'q2_cost_only/planning_day{day}.npz')
            np.testing.assert_array_equal(src['selected_scenario_indices'], own['selected_scenario_indices'])
            rows.append({'day': day, 'slot': 0, 'matched': True,
                         'net_paths_sha256': array_sha(src['all_net_paths']),
                         'price_sha256': array_sha(src['price']), 'frozen_source_sha256': digest(source)})
        issued = Q2OLD/'issued_forecasts.npz'
        sources = [ref(issued), ref(NEW/'q2_cost_only/provenance.json')]
        assert len(rows) == 334
        scope = 'Exact immutable exogenous input archives and identical forecast/history audit; no old endogenous decisions are inputs.'
    elif scenario == '4-2':
        directory = NEW/'q4_2_cost_only'
        issued, baseline = load(directory/'issued_forecasts.npz'), load(Q4OLD/'issued_forecasts.npz')
        for key in ('values', 'origins'):
            np.testing.assert_array_equal(issued[key], baseline[key])
        protocol = read(directory/'protocol.json')
        assert protocol['issued_values_sha256'] == array_sha(issued['values'])
        assert protocol['issued_price_sha256'] == array_sha(issued['price'])
        assert protocol['global_protocol_sha256'] == digest(PROTOCOL)
        for i, new in enumerate(after):
            day = 31+i
            oldpath, newpath = (base/f'daily_chunks/planning_day_{day}.npz' for base in (Q4OLD, directory))
            old, own = load(oldpath), load(newpath)
            for key in ('all_net_paths', 'predicted_price', 'selected_scenario_indices', 'all_price_error_paths'):
                np.testing.assert_array_equal(old[key], own[key])
            np.testing.assert_array_equal(own['load_kw'], issued['values'][i, :, 0])
            np.testing.assert_array_equal(own['pv_kw'], issued['values'][i, :, 1])
            np.testing.assert_array_equal(own['predicted_price'], issued['price'][i])
            for key, expected in new['issued_array_sha256'].items():
                assert array_sha(own[key]) == expected
            rows.append({'day': day, 'slot': 0, 'matched': True, **new['issued_array_sha256']})
        sources = [ref(directory/'protocol.json'), ref(directory/'issued_forecasts.npz'), ref(Q4OLD/'issued_forecasts.npz')]
        assert len(rows) == 334
        scope = 'All midnight load/PV/price values and all historical input paths exactly equal frozen exp008 arrays.'
    else:
        assert len(before) == len(after) == 1336
        for i, (old, new) in enumerate(zip(before, after, strict=True)):
            day, slot = 31+i//4, (i%4)*36
            assert (old['day'], old['slot']) == (new['day'], new['slot']) == (day, slot)
            for key in ('forecast', 'residual_paths', 'issued_prediction_sha256', 'information_cutoff',
                        'training_origins', 'max_observed_index', 'original_locked', 'executed_until'):
                assert old[key] == new[key], (scenario, day, slot, key)
            assert not new['forecast']['known_future_price']
            assert new['residual_paths']['label_stops_exclusive'][-1] <= day*144
            rows.append({'day': day, 'slot': slot, 'matched': True, **new['issued_prediction_sha256'],
                'net_paths_sha256': new['residual_paths']['errors_kwh_sha256'],
                'price_paths_sha256': new['residual_paths']['price_errors_sha256']})
        a, b = (read(base/scenario/'completion.json')['config']['forecast_override'] for base in (UPO, UPN))
        a, b = dict(a), dict(b)
        module_a, module_b = a.pop('implementation_class'), b.pop('implementation_class')
        assert module_a.rsplit('.', 1)[-1] == module_b.rsplit('.', 1)[-1] == 'LinkedHGBForecasts'
        assert a == b
        sources = [ref(base/scenario/name) for base in (UPO, UPN) for name in ('audit.json', 'completion.json')]
        scope = 'All1336 issued load/PV/price and same-issued residual hashes equal; only descriptive __main__ module qualifier differs.'
    assert all(row['matched'] for row in rows)
    pd.DataFrame(rows).to_csv(OUT/f'q{scenario.replace("-", "_")}_forecast_identity.csv', index=False)
    return {'passed': True, 'matched_issues': len(rows), 'scope': scope, 'sources': sources,
        'future_mutation_test_not_repeated': True,
        'causality_scope': 'Identity with exp008 causal issue evidence and cutoff checks; this script does not independently rerun forecasters.'}


def checked_arrays(a, scenario, initial_soc, initial_power, actual, price, audits):
    result = verify_arrays(a, scenario, expected_days=1 if scenario == '1' else 334,
        start_day=0 if scenario == '1' else 31, initial_soc=initial_soc,
        initial_mode=0 if scenario == '1' else 1, initial_power_kw=initial_power,
        source_actual=actual, source_price=price, audit_records=audits)
    # Old exp006 acceptance goals are unrelated to this controlled treatment.
    result.pop('goal', None)
    assert result['passed'], result['errors']
    # Second direct calculation uses raw CSV values, not saved fees/actual.
    c, d, soc = a['charge'], a['discharge'], a['states']
    balance = a['final'] + actual[..., 1]/6 + d + a['emergency'] - actual[..., 0]/6 - c - a['surplus']
    transition = soc[:, 1:]-soc[:, :-1]-ETA*c+d/ETA
    bill = price*a['original'] + 1.5*price*np.maximum(a['final']-a['original'], 0) + .5*price*np.maximum(a['original']-a['final'], 0) + 5*price*a['emergency']
    assert np.max(np.abs(balance)) < TOL and np.max(np.abs(transition)) < TOL
    assert abs(float(bill.sum())-result['recomputed_total_cost']) < 1e-6
    if scenario != '1':
        pre_battery = a['final']+(actual[..., 1]-actual[..., 0])/6
        assert not np.any((d > TOL) & (pre_battery > TOL))
        assert not np.any((c > TOL) & (pre_battery < -TOL))
        np.testing.assert_allclose(a['emergency'], np.maximum(-pre_battery-d, 0), rtol=0, atol=TOL)
        np.testing.assert_allclose(a['surplus'], np.maximum(pre_battery-c, 0), rtol=0, atol=TOL)
    if 'allowed_charge' in a:
        mask = np.asarray(a['allowed_charge'], bool)
        assert mask.shape == c.shape and not np.any(c[~mask] > TOL) and not np.any(d[mask] > TOL)
        result['actual_actions_follow_fixed_mask'] = True
    result['second_raw_csv_calculation'] = {'passed': True, 'total_cost_yuan': float(bill.sum()),
        'max_balance_error_kwh': float(np.abs(balance).max()), 'max_soc_error_kwh': float(np.abs(transition).max())}
    return result


def export_power(a, check, scenario, experiment, initial_power, extra=None):
    directory = OUT/f'q{scenario.replace("-", "_")}'/experiment
    directory.mkdir(parents=True, exist_ok=True)
    power = 6*(a['charge']-a['discharge']).ravel()
    delta = np.diff(power)
    endpoints = pd.date_range('2025-02-01 00:10', periods=len(power), freq='10min')
    frame = pd.DataFrame({'interval_start': endpoints-pd.Timedelta(minutes=10), 'interval_end': endpoints,
        'charge_power_kw': 6*a['charge'].ravel(), 'discharge_power_kw': 6*a['discharge'].ravel(),
        'net_battery_power_kw': power, 'delta_power_kw': np.r_[np.nan, delta],
        'soc_start_kwh': a['states'][:, :-1].ravel(), 'soc_end_kwh': a['states'][:, 1:].ravel(),
        'load_kw': a['actual'][..., 0].ravel(), 'pv_kw': a['actual'][..., 1].ravel()})
    if scenario == '1':
        # Attachment1 has no dated annual trajectory; don't invent a date.
        frame = frame.drop(columns=['interval_start', 'interval_end'])
        frame.insert(0, 'interval_start_hour', np.arange(144)/6)
        frame.insert(1, 'interval_end_hour', np.arange(1, 145)/6)
    frame.to_csv(directory/'power_all_intervals.csv', index=False)
    if scenario != '1':
        assert len(frame) == 48096
        for day in np.unique(np.r_[RANDOM, SPECIFIED_IDX]):
            frame.iloc[day*144:(day+1)*144].to_csv(directory/f'power_{DATES[day]:%Y-%m-%d}.csv', index=False)
    b = check['battery_metrics']
    metrics = {'scenario': scenario, 'experiment': experiment, 'model_seed': None if scenario == '1' else 42,
        'days': len(a['charge']), 'points': len(power), 'cost_yuan': check['recomputed_total_cost'],
        **check['billing'], **b, 'episodes': b['charge_starts']+b['discharge_starts'],
        'mean_absolute_delta_kw': float(np.abs(delta).mean()), 'rms_delta_kw': float(np.sqrt(np.mean(delta**2))),
        'p95_absolute_delta_kw': float(np.quantile(np.abs(delta), .95)), 'max_absolute_delta_kw': float(np.abs(delta).max()),
        'total_variation_kw': float(np.abs(delta).sum()),
        'direct_reversals': int(np.sum(((power[:-1]>TOL)&(power[1:]<-TOL))|((power[:-1]<-TOL)&(power[1:]>TOL)))),
        'power_limit_share': float(np.mean(np.abs(power)>=5000-TOL)),
        'large_jump_share': float(np.mean(np.abs(delta)>1000+TOL)),
        'warmup_boundary_delta_kw': None if scenario == '1' else float(power[0]-initial_power),
        'extra_q1_cyclic_metrics': extra}
    if extra is not None:
        metrics.update({key: extra[key] for key in ('cyclic_total_variation_kw', 'cyclic_mode_starts',
                        'cyclic_min_on_minutes', 'cyclic_min_off_minutes', 'max_cyclic_step_kw')})
    save(directory/'power_metrics.json', metrics)
    reread = pd.read_csv(directory/'power_all_intervals.csv')
    np.testing.assert_allclose(reread.net_battery_power_kw, power, rtol=0, atol=1e-8)
    assert reread.delta_power_kw.isna().sum() == 1 and pd.isna(reread.delta_power_kw.iloc[0])
    return frame, metrics


def style():
    path = Path('/System/Library/Fonts/STHeiti Light.ttc')
    if path.exists():
        fm.fontManager.addfont(str(path))
        matplotlib.rcParams['font.family'] = fm.FontProperties(fname=str(path)).get_name()
    matplotlib.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#51545a',
        'text.color': '#30343b', 'path.simplify': False, 'agg.path.chunksize': 0,
        'svg.fonttype': 'none', 'svg.hashsalt': 'exp009-battery', 'savefig.dpi': 180})


def axis(ax):
    ax.axhline(0, color='#52565d', lw=.7)
    for p in (-5000, 5000):
        ax.axhline(p, color='#a8abb0', ls=':', lw=.7)
    ax.set_ylim(-5450, 5450)
    ax.set_yticks([-5000, 0, 5000])
    ax.set_ylabel('净电池功率 / kW')
    ax.grid(axis='y', color='#e5e7eb', lw=.5)
    ax.spines[['top', 'right']].set_visible(False)


def store_figure(fig, name, scenario, kind, points, ledger):
    for extension in ('png', 'svg'):
        path = FIG/f'{name}.{extension}'
        fig.savefig(path, bbox_inches='tight', facecolor='white',
                    **({'metadata': {'Date': None}} if extension == 'svg' else {}))
        path_points = None
        if extension == 'svg':
            path_points = []
            for element in ET.fromstring(path.read_text()).iter('{http://www.w3.org/2000/svg}path'):
                styling = element.attrib.get('style', '')
                if any('stroke: '+color in styling for color in COLORS.values()):
                    count = len(re.findall(r'\b[ML]\s', element.attrib.get('d', '')))
                    if count > 10:
                        path_points.append(count)
            expected = [points]*(8 if kind in ('random', 'specified') else 2)
            assert path_points == expected, (name, path_points, expected)
        ledger.append({**ref(path), 'scenario': scenario, 'kind': kind,
            'series': ['exp008', 'exp009'], 'points_per_series_per_panel': points,
            'svg_serialized_path_point_counts': path_points,
            'smoothing': False, 'downsampling': False, 'path_simplification': False})
    plt.close(fig)


def plot_pair(scenario, frames, ledger):
    slug = 'q'+scenario.replace('-', '_')
    if scenario == '1':
        fig, ax = plt.subplots(figsize=(14, 4.8))
        for label, df in frames.items():
            line = ax.plot(df.interval_end_hour, df.net_battery_power_kw, color=COLORS[label],
                           ls='--' if label == 'exp008' else '-', lw=1.2, label=label)[0]
            assert len(line.get_xdata()) == 144
        axis(ax); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel('区间终点 / 小时'); ax.legend(frameon=False)
        ax.set_title('第一问：费用单目标与 exp008 的确定性日计划（各144点）', loc='left')
        fig.text(.08, .01, '充电为正、放电为负。附件1的一日计划；不标为年度实测。ramp与最短启停工程假设固定。', fontsize=9)
        fig.subplots_adjust(bottom=.18)
        store_figure(fig, slug+'_deterministic_144_points', scenario, 'deterministic_day', 144, ledger)
        return
    for kind, indices in (('random', RANDOM), ('specified', SPECIFIED_IDX)):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharex=True, sharey=True)
        for ax, day in zip(axes.flat, indices):
            for label, df in frames.items():
                values = df.iloc[day*144:(day+1)*144].net_battery_power_kw
                line = ax.plot(np.arange(1, 145)/6, values, color=COLORS[label],
                    ls='--' if label == 'exp008' else '-', lw=1.05, label=label)[0]
                assert len(line.get_xdata()) == 144
            axis(ax); ax.set_title(f'{DATES[day]:%Y-%m-%d}', loc='left')
            ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24], ['00:00', '06:00', '12:00', '18:00', '24:00'])
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
        fig.suptitle(f'问题{scenario}：完整十分钟实际执行功率', x=.07, ha='left', fontsize=15)
        subtitle = f'固定种子{SEED}，不放回抽样后排序' if kind == 'random' else '题目指定四日'
        fig.text(.07, .02, f'{subtitle}；充电为正、放电为负。每条每日144点；00:10代表00:00–00:10区间终点。', fontsize=9)
        fig.subplots_adjust(top=.86, bottom=.10, hspace=.27, wspace=.15)
        store_figure(fig, slug+'_'+kind+'_days', scenario, kind, 144, ledger)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True, sharey=True)
    for ax, (label, df) in zip(axes, frames.items()):
        line = ax.plot(pd.to_datetime(df.interval_end), df.net_battery_power_kw, color=COLORS[label], lw=.25)[0]
        assert len(line.get_xdata()) == 48096
        axis(ax); ax.set_title(label, loc='left')
    ticks = pd.to_datetime(['2025-02-01', '2025-04-01', '2025-06-01', '2025-08-01', '2025-10-01', '2026-01-01'])
    axes[-1].set_xticks(ticks, ['02-01', '04-01', '06-01', '08-01', '10-01', '12-31 24:00'])
    axes[-1].set_xlim(pd.Timestamp('2025-02-01'), pd.Timestamp('2026-01-01'))
    fig.suptitle(f'问题{scenario}：334日正式评价，各48,096个原始点', x=.065, ha='left', fontsize=15)
    fig.text(.065, .025, '2025-02-01至2025-12-31；预热不拼入评分。无抽稀、无平滑；SVG与CSV保留所有十分钟时点。', fontsize=9)
    fig.subplots_adjust(top=.88, bottom=.12, hspace=.29)
    store_figure(fig, slug+'_all_48096_points', scenario, 'annual', 48096, ledger)


def ready(scenario):
    _, path, marker, _ = CASES[scenario]
    if not path.exists() or (marker is not None and not marker.exists()):
        return False
    if marker is not None and read(marker).get('complete') is not True:
        return False
    if scenario != '1':
        with np.load(path, allow_pickle=False) as z:
            return z['charge'].shape == (334, 144) and z['states'].shape == (334, 145)
    return len(read(path)['stages']) == 1


def run(require_all=False, selected=None):
    began = perf_counter()
    selected = list(CASES) if selected is None else selected
    pending = [s for s in selected if not ready(s)]
    if require_all and pending:
        raise FileNotFoundError('Final annual archives not complete: '+', '.join(pending))
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    actual, q1actual, prices, raw = raw_sources()
    baseline_random = ROOT/'reports/experiments/exp008/evidence/battery/q2_final/manifest.json'
    assert read(baseline_random)['random_dates'] == [str(DATES[i].date()) for i in RANDOM]
    style(); cases, rows, comparisons, ledger = {}, [], [], []
    for scenario in selected:
        if scenario in pending:
            cases[scenario] = {'complete': False, 'status': 'pending', 'annual_cost': None, 'annual_battery_metrics': None}
            continue
        oldpath, newpath, marker, warmup = CASES[scenario]
        if warmup:
            w = load(warmup)
            soc = float(w['states'][-1, -1]); power = float(6*(w['charge'][-1, -1]-w['discharge'][-1, -1]))
        else:
            soc, power = 6000., 0.
        assert soc == read(PROTOCOL)['fixed']['initial_soc_kwh'][scenario]
        if scenario != '1':
            assert power == read(PROTOCOL)['fixed']['annual_initial_power_kw'] and np.sign(power) == 1
        source_actual = q1actual if scenario == '1' else actual
        price = prices['realized'] if scenario.startswith('4') else prices['fixed']
        frames, checks, metrics, audits = {}, {}, {}, []
        for experiment, path in (('exp008', oldpath), ('exp009', newpath)):
            if scenario == '1':
                a, extra = q1_arrays(path, q1actual, prices['fixed']); audit = None
            else:
                a, extra = load(path), None
                auditpath = path.parent/AUDITS[scenario][0 if experiment == 'exp008' else 1]
                audit = read(auditpath); audits.append(audit)
            check = checked_arrays(a, scenario, soc, power, source_actual, price, audit)
            frame, metric = export_power(a, check, scenario, experiment, power, extra)
            save(OUT/f'q{scenario.replace("-", "_")}'/experiment/'manifest.json', {
                'source': ref(path), 'verification': check, 'warmup_source': ref(warmup) if warmup else None,
                'initial_state_matches_verified_warmup': bool(warmup), 'complete': True,
                'scope': 'deterministic single day' if scenario == '1' else '334days / 48096points; warmup excluded',
                'q1_specialized_checks': extra, 'power_csv_verified_against_arrays': True})
            frames[experiment], checks[experiment], metrics[experiment] = frame, check, metric
            rows.append({k: v for k, v in metric.items() if not isinstance(v, (dict, list))})
        identity = ({'passed': True, 'scope': 'Same144-row attachment1 forecast input; no learned predictor'} if scenario == '1'
                    else forecast_identity(scenario, audits[0], audits[1]))
        cases[scenario] = {'complete': True, 'passed': True, 'completion_marker': ref(marker) if marker else ref(newpath),
            'forecast_identity': identity, 'verification': checks, 'battery_metrics': metrics,
            'primary_endpoint': 'actual total fees; counts are descriptive, with no improvement gate'}
        for key, before in metrics['exp008'].items():
            after = metrics['exp009'].get(key)
            if key in ('model_seed', 'days', 'points', 'initial_mode', 'initial_power_kw'):
                continue
            if isinstance(before, (int, float)) and not isinstance(before, bool) and isinstance(after, (int, float)):
                comparisons.append({'scenario': scenario, 'metric': key, 'previous': before, 'current': after,
                    'absolute_change': after-before, 'relative_change_pct': 100*(after-before)/abs(before) if before else None,
                    'scope': 'same raw data, rated hardware, boundary state and actual billing; stated auxiliary treatment differs'})
        plot_pair(scenario, frames, ledger)
        print(json.dumps({'scenario': scenario, 'complete': True, 'independent_checks_passed': True}, ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT/'battery_metrics_all.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'exp009_vs_exp008_comparison.csv', index=False)
    visual = {'status': 'pending separate rendered inspection'}
    review_path = OUT/'visual_review.json'
    if review_path.exists():
        review = read(review_path)
        expected_pngs = {row['path'] for row in ledger if row['path'].endswith('.png')}
        reviewed_pngs = {row['path'] for row in review['reviewed_pngs']}
        if review['passed'] and expected_pngs == reviewed_pngs and all(
                digest(ROOT/row['path']) == row['sha256'] for row in review['reviewed_pngs']):
            visual = {'status': 'passed', 'separate_manual_review': ref(review_path)}
    result = {'complete': not pending and set(selected) == set(CASES), 'pending': pending,
        'selected_cases': selected, 'source': ref(Path(__file__)), 'protocol': ref(PROTOCOL),
        'raw_data_validation': raw, 'cases': cases, 'figures': ledger,
        'random_seed': SEED, 'random_day_algorithm': 'numpy.random.default_rng(seed).choice(334,4,replace=False); sort',
        'random_dates': [str(DATES[i].date()) for i in RANDOM], 'specified_dates': [str(x.date()) for x in SPECIFIED],
        'metric_definitions': {'formal_delta': 'np.diff over the continuous formal trace, including midnight; first point missing',
            'warmup_delta': 'first formal power minus warmup last power, separate from formal statistics',
            'nonidle_reversals': 'opposite consecutive nonzero directions after removing idle; energy tolerance1e-6kWh',
            'direct_reversals': 'opposite adjacent nonzero powers, without crossing idle; power tolerance1e-6kW',
            'episodes': 'charge starts+discharge starts, each contiguous signed activity run; no merging across idle',
            'large_jump': '|delta power|>1000+1e-6kW, descriptive threshold not an added annual hard constraint',
            'EFC': '(sqrt(.9)*ACcharge+ACdischarge/sqrt(.9))/(2*12000), intensity proxy not calibrated ageing'},
        'no_smoothing_or_downsampling': True, 'figure_paths_verified_point_counts': True,
        'visual_review': visual, 'wall_seconds': perf_counter()-began}
    save(OUT/'final_evidence.json', result)
    save(OUT/'figure_manifest.json', ledger)
    print(json.dumps({'complete': result['complete'], 'pending': pending, 'figure_files': len(ledger),
                      'evidence': str(OUT.relative_to(ROOT))}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-all', action='store_true')
    parser.add_argument('--cases', nargs='+', choices=list(CASES))
    args = parser.parse_args()
    run(args.require_all, args.cases)
