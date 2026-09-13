"""Build the eight-section report from the continuous turnover-penalty runs."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from evidence import BASE, OUT as EVIDENCE, ROOT, collect, read

sys.path.insert(0, str(ROOT))
from reports.experiments.exp005.strict_rerun.prepare_workbook import clock, periods  # noqa: E402

REPORT = ROOT / 'reports/experiments/exp005'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def rows(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    frames = collect()
    audit_labels = {'balance': '逐段供需平衡残差 / kWh', 'soc': 'SOC递推残差 / kWh',
        'bounds': '电量与SOC边界超出 / kWh', 'fixed_grid': '日内普通计划变化 / kWh',
        'actual_source': '实际输入与冻结原值差 / kW', 'cross_day_soc': '跨日SOC断点 / kWh',
        'ramp': '相邻净功率最大变化 / kW', 'node_constraints': '规划节点最大约束残差',
        'cost_budget': '第二层超预算金额 / 元', 'fees': '独立账单差额 / 元', 'integer_variables': '整数变量数'}
    frames['audit']['check_label'] = frames['audit']['check'].map(audit_labels)
    daily, actual, runs = (frames[k] for k in ['daily', 'actual', 'runs'])
    main_daily = daily[daily.beta == .1].copy()
    main_actual = actual[actual.beta == .1].copy()
    n, last = len(main_daily), main_daily.date.iloc[-1]
    finished = len(runs) == 2 and all(runs.status == 'complete') and all(runs.completed_days == 334)
    if args.complete and not finished:
        raise RuntimeError('Both approved 334-day replays must finish before final report completion')
    baseline = dict(np.load(BASE / 'exp004_dispatch_2.npz'))
    common = int(runs.completed_days.min())
    common_rows = []
    period_rows = []
    for beta, sub in daily.groupby('beta'):
        label = f'β={beta:g}' + (' 主方案' if beta == .1 else '敏感性对照')
        matched = sub.iloc[:common]
        record = {'label': label, 'beta': beta, 'days': len(sub), 'total_cost': float(sub.total_cost.sum()),
                  'planned_cost': float(sub.planned_cost.sum()), 'emergency_cost': float(sub.emergency_cost.sum()),
                  'adjustment_cost': 0., 'emergency_kwh': float(sub.emergency_kwh.sum()),
                  'overlap_intervals': int(sub.overlap_intervals.sum()), 'max_overlap_kwh': float(sub.max_overlap_kwh.max()),
                  'throughput_kwh': float(sub.throughput_kwh.sum()), 'tv_kw': float(sub.tv_kw.sum())}
        period_rows.append(dict(record, total_wan=record['total_cost'] / 10000))
        common_rows.append({'label': label, 'beta': beta, 'days': common, 'last_date': matched.date.iloc[-1],
             **{k: float(matched[k].sum()) for k in ['total_cost', 'planned_cost', 'emergency_cost', 'throughput_kwh', 'tv_kw']},
             'overlap_intervals': int(matched.overlap_intervals.sum())})
    base_cost = baseline['fees'][:n].sum(axis=(0, 1))
    period_rows.insert(0, {'label': 'exp004 无季节·原调度器', 'beta': None, 'days': n,
         'total_cost': float(base_cost.sum()), 'planned_cost': float(base_cost[0]), 'emergency_cost': float(base_cost[-1]),
         'adjustment_cost': float(base_cost[1:-1].sum()), 'emergency_kwh': float(baseline['emergency'][:n].sum()),
         'total_wan': float(base_cost.sum() / 10000)})
    frames['period'] = pd.DataFrame(period_rows)
    np.testing.assert_allclose(frames['period'].total_cost,
        frames['period'].planned_cost + frames['period'].emergency_cost + frames['period'].adjustment_cost, rtol=0, atol=1e-6)
    frames['paired_beta'] = pd.DataFrame(common_rows)
    frames['monthly'] = daily.assign(month=daily.date.str[:7]).groupby(['beta', 'month'], as_index=False).agg(
         days=('date', 'count'), total_cost=('total_cost', 'sum'), emergency_cost=('emergency_cost', 'sum'),
         overlap_intervals=('overlap_intervals', 'sum'), throughput_kwh=('throughput_kwh', 'sum'))
    hourly = actual.assign(hour_band=actual.slot // 6).groupby(['beta', 'month', 'hour_band'], as_index=False).agg(
         intervals=('slot', 'count'), overlap_intervals=('overlap_flag', 'sum'), overlap_kwh=('significant_overlap_kwh', 'sum'))
    hourly['overlapRate'] = hourly.overlap_intervals / hourly.intervals
    hourly['hour_label'] = hourly.hour_band.map(lambda x: f'{x:02}:00')
    frames['overlap_heatmap'] = hourly
    scope_map = {'midnight': '午夜规划', 'execution': '日内滚动'}
    frames['overlap_summary']['scope_label'] = frames['overlap_summary'].apply(
        lambda r: f'{scope_map[r.scope]}·第{int(r.layer)}层', axis=1)
    frames['stage_daily']['scope_label'] = frames['stage_daily'].apply(
        lambda r: f'{scope_map[r.scope]}·第{int(r.layer)}层', axis=1)
    first_second = frames['stage_daily'].groupby(['beta', 'date', 'layer'], as_index=False).agg(
        max_overlap_kwh=('max_overlap_kwh', 'max'), overlap_nodes=('overlap_nodes', 'sum'), nodes=('nodes', 'sum'))
    first_second['label'] = first_second.layer.map({1: '第一层参考解', 2: '第二层方案'})
    frames['overlap_timeline'] = first_second
    residual = frames['overlap_nodes'].query('beta == .01 and layer == 2')
    frames['control_residual_case'] = pd.DataFrame()
    if len(residual):
        worst = residual.loc[residual.overlap_kwh.idxmax()]
        prefix = ROOT / f'data/results/exp005/soft-penalty-beta-0.01/{worst.date}/largest_second_layer_overlap'
        raw = dict(np.load(str(prefix) + '.npz'))
        info = read(Path(str(prefix) + '.json'))
        node = int(np.argmin(np.abs(np.minimum(raw['charge'], raw['discharge']) - worst.overlap_kwh)))
        chain = [node]
        while raw['parent'][chain[0]] >= 0:
            chain.insert(0, int(raw['parent'][chain[0]]))
        while len(children := np.flatnonzero(raw['parent'] == chain[-1])):
            chain.append(int(children[0]))
        frames['control_residual_case'] = pd.DataFrame([{'date': worst.date, 'replay_slot': info['slot'],
            'node': j, 'hour': (info['slot'] + int(raw['time'][j])) / 6,
            'charge_kwh': raw['charge'][j], 'discharge_kwh': raw['discharge'][j],
            'overlap_kwh': min(raw['charge'][j], raw['discharge'][j]), 'probability': raw['probability'][j]}
            for j in chain if abs(info['slot'] + int(raw['time'][j]) - worst.future_slot) <= 18])
    preflight = read(ROOT / 'data/results/exp005/soft-penalty-preflight/results.json')
    pre_rows, pre_curves = [], []
    for r in preflight:
        if r['window'] != 'original-lp-overlap' or r['beta'] not in [0, .01, .1]:
            continue
        meta = r['metadata']
        pre_rows.append({'beta': r['beta'], 'label': f'β={r["beta"]:g}', 'max_overlap_kwh': r['max_overlap_kwh'],
                        'expected_tv_kw': meta['expected_tv_kw'], 'second_cost': meta['second_cost'],
                        'expected_throughput_kwh': r['expected_throughput_kwh']})
        z = dict(np.load(ROOT / f'data/results/exp005/soft-penalty-preflight/original-lp-overlap-beta-{r["beta"]}.npz'))
        for t in range(len(z['charge'])):
            pre_curves.append({'beta': r['beta'], 'hour': (81 + t) / 6, 'charge_kwh': float(z['charge'][t]),
                               'discharge_kwh': float(z['discharge'][t]), 'overlap_kwh': float(min(z['charge'][t], z['discharge'][t]))})
    frames['preflight'] = pd.DataFrame(pre_rows)
    frames['preflight_curves'] = pd.DataFrame(pre_curves)
    history = pd.read_csv(BASE / 'history.csv')
    history = history[~history.label.str.startswith('exp005')].copy()
    for r in period_rows[1:]:
        history.loc[len(history)] = {'label': f'exp005 周转惩罚 {r["label"]}', 'total_cost': r['total_cost'],
              'planned_cost': r['planned_cost'], 'emergency_cost': r['emergency_cost'],
              'status': '新增硬爬坡、场景、两层滚动及周转惩罚，费用描述性并列',
              'period': f'2025-02-01—{daily[daily.beta == r["beta"]].date.iloc[-1]}（{r["days"]}天）'}
    strict = pd.DataFrame(read(ROOT / 'data/results/exp005/strict-mutual-exclusion-certified/daily.json'))
    history.loc[len(history)] = {'label': 'exp005 严格互斥·已暂停', 'total_cost': strict.total_cost.sum(),
             'planned_cost': strict.planned_cost.sum(), 'emergency_cost': strict.emergency_cost.sum(),
             'status': '原零差距MILP；55日，不能与334日总额排名', 'period': f'2025-02-01—{strict.date.iloc[-1]}（{len(strict)}天）'}
    frames['history'] = history
    forecast_history = pd.read_csv(BASE / 'forecast_history.csv')
    forecast_history['label'] = forecast_history.label.replace({'exp005 沿用同一冻结预测': 'exp005 周转惩罚·同一冻结预测'})
    frames['forecast_history'] = forecast_history
    with np.load(ROOT / 'data/results/exp004/predictions.npz') as z:
        forecasts = z['no_season_seed_42'].copy()
    predictions = np.dstack([forecasts, forecasts[:, :, 0] - forecasts[:, :, 1]])
    observations = np.dstack([baseline['actual'], baseline['actual'][:, :, 0] - baseline['actual'][:, :, 1]])
    forecast_rows = []
    full_dates = pd.date_range('2025-02-01', periods=334)
    for target, name in enumerate(['load', 'pv', 'net_load']):
        for month in [0, *range(2, 13)]:
            take = np.ones(334, dtype=bool) if month == 0 else full_dates.month == month
            for population in (['all', 'actual_generation'] if name == 'pv' else ['all']):
                p, a = predictions[take, :, target], observations[take, :, target]
                mask = np.ones(a.shape, dtype=bool) if population == 'all' else a > 0
                error = (p - a)[mask]
                denom = np.abs(a[mask]).sum()
                forecast_rows.append({'target': name, 'month': month, 'population': population,
                   'n': len(error), 'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error ** 2).mean())),
                   'wape_pct': float(100 * np.abs(error).sum() / denom) if denom > 0 else None})
        reference = forecast_history[(forecast_history.label == 'exp004 无季节') & (forecast_history.target == name)].iloc[0]
        annual = next(r for r in forecast_rows if r['target'] == name and r['month'] == 0 and r['population'] == 'all')
        for metric in ['mae', 'rmse', 'wape_pct']:
            np.testing.assert_allclose(annual[metric], reference[metric], atol=1e-8, rtol=0)
    frames['forecast_breakdown'] = pd.DataFrame(forecast_rows)
    timing = pd.read_csv(BASE / 'timing_history.csv')
    timing = timing[timing.experiment != 'exp005'].copy()
    for r in runs.to_dict('records'):
        for stage, seconds in [('新增训练', 0.), ('回放墙钟', r['wall_seconds']), ('求解器累计', r['solver_seconds'])]:
            timing.loc[len(timing)] = {'experiment': 'exp005', 'label': f'β={r["beta"]:g}', 'stage': stage,
                 'seconds': seconds, 'groups': r['completed_days'],
                 'scope': f'{r["completed_days"]}日；两组顺序运行，未重训。求解时间与回放时间有包含关系，不相加。',
                 'source': f'data/results/exp005/soft-penalty-beta-{r["beta"]}/status.json'}
    frames['timing_history'] = timing
    frames['timing_plot'] = timing[timing.stage.isin(['训练', '新增训练', '调度执行', '回放墙钟'])].dropna(subset=['seconds']).copy()
    frames['timing_plot']['item'] = frames['timing_plot'].label + ' · ' + frames['timing_plot'].stage
    frames['technical_comparison'] = pd.DataFrame([
        {'scheme': 'exp004 无季节', 'forecast': '同一冻结午夜预测', 'midnight': '确定性午夜MILP', 'execution': '日内贪心反馈', 'ramp': '未采用本轮硬爬坡协议', 'overlap': '原控制器充放电逻辑'},
        {'scheme': 'exp005 严格互斥旧版', 'forecast': '同一冻结午夜预测', 'midnight': '场景信息树、两层MILP', 'execution': '逐十分钟滚动MILP', 'ramp': '含跨日1000 kW硬爬坡', 'overlap': '0/1互斥；已暂停55日'},
        {'scheme': 'exp005 周转惩罚', 'forecast': '同一冻结午夜预测', 'midnight': '场景信息树、两层LP', 'execution': '逐十分钟滚动LP', 'ramp': '含跨日1000 kW硬爬坡', 'overlap': '第二层β(c+d)归一化惩罚；事后诊断'}])
    comparison_rows = []
    for beta in [.1, .01]:
        item = next(r for r in common_rows if r['beta'] == beta)
        reference = next(r for r in common_rows if r['beta'] == .01)
        for metric in ['total_cost', 'throughput_kwh', 'tv_kw', 'overlap_intervals']:
            previous, current = reference[metric], item[metric]
            comparison_rows.append({'comparison': f'β={beta:g} 对 β=0.01', 'metric': metric,
                'days': common, 'previous': previous, 'current': current, 'absolute_change': current - previous,
                'relative_change_pct': 100 * (current - previous) / abs(previous) if previous else None,
                'comparable': True, 'scope': '相同预测、场景和初态；各自连续执行，只有β不同'})
    frames['metric_comparison'] = pd.DataFrame(comparison_rows)
    specified = {k: [] for k in ['specified_plan', 'specified_battery', 'specified_soc', 'specified_emergency']}
    for day in ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']:
        if day not in set(main_daily.date):
            continue
        z = dict(np.load(ROOT / f'data/results/exp005/soft-penalty-beta-0.1/{day}/actual.npz'))
        for h in [10, 12, 14, 16, 18, 20]:
            specified['specified_plan'].append({'date': day, 'interval': f'{h}:00—{h}:10', 'grid_kwh': float(z['grid'][h * 6])})
        for b in range(6):
            specified['specified_battery'].append({'date': day, 'interval': f'{clock(b * 24)}—{clock((b + 1) * 24)}',
                 'charge_kwh': float(z['charge'][b * 24:(b + 1) * 24].sum()),
                 'discharge_kwh': float(z['discharge'][b * 24:(b + 1) * 24].sum())})
        specified['specified_soc'].append({'date': day, 'initial_soc_kwh': float(z['states'][0]), 'final_soc_kwh': float(z['states'][-1])})
        for a, b, value in periods(z['emergency']) or [(0, 0, 0.)]:
            specified['specified_emergency'].append({'date': day, 'interval': f'{clock(a)}—{clock(b)}' if b else '无', 'emergency_kwh': value})
    frames.update({k: pd.DataFrame(v) for k, v in specified.items()})
    total, base_total = float(main_daily.total_cost.sum()), float(base_cost.sum())
    change = total - base_total
    observed_overlap = int(main_actual.overlap_flag.sum())
    second_nodes = frames['overlap_summary'].query('beta == 0.1 and layer == 2')
    first_nodes = frames['overlap_summary'].query('beta == 0.1 and layer == 1')
    control_nodes = frames['overlap_summary'].query('beta == 0.01 and layer == 2')
    paired = frames['paired_beta'].set_index('beta')
    beta_difference = paired.loc[.1, 'total_cost'] - paired.loc[.01, 'total_cost']
    beta_text = (f'共同{common}日内，β=0.1相比β=0.01费用变化{beta_difference:+,.4f}元'
        f'（{100 * beta_difference / paired.loc[.01, "total_cost"]:+.4f}%），'
        f'充放电周转量变化{paired.loc[.1, "throughput_kwh"] - paired.loc[.01, "throughput_kwh"]:+,.4f} kWh，'
        f'净功率总变差变化{paired.loc[.1, "tv_kw"] - paired.loc[.01, "tv_kw"]:+,.4f} kW。'
        f'两组实际重叠分别为{int(paired.loc[.1, "overlap_intervals"])}段与{int(paired.loc[.01, "overlap_intervals"])}段。'
        '这是各自状态连续演化后的全年策略比较；更高权重不保证实际全年每项指标单调改善。')
    verification_path = ROOT / 'data/results/exp005/verification.json'
    workbook_ready = verification_path.exists() and read(verification_path).get('result2.xlsx', {}).get('saved_workbook_readback') == 'passed'
    workbook_text = ('主方案完整购电策略已导出并回读核验：[result2.xlsx](result2.xlsx)。含334日的144段普通购电、四小时充放电、日初日末SOC及紧急购电区间；β=0.01作为报告中的敏感性对照。'
                     if workbook_ready else '完整Excel在主方案334日完成并核验后生成。')
    title = 'exp005：周转惩罚与充放电重叠检验'
    progress_text = ('两组均已完成2025-02-01至12-31的334日回放。' if finished else
                     '回放进行中：' + '；'.join(f'β={r.beta:g}完成{r.completed_days}日，截至{r.last_date}' for r in runs.itertuples()) + '。未完成日期不填零。')
    penalty_formula = 'J₂ = Σqₙ|Pₙ−P父节点|/(2PmaxT) + βΣqₙ(cₙ+dₙ)/(2PmaxΔtT)'
    sections = [
        ('结论与成绩', f'## 1. 结论与成绩\n\n{progress_text}\n\n本次按用户决定移除整数开关，在第二层加入归一化周转惩罚。主方案β=0.1，敏感性对照β=0.01；原实验源码未修改。β在回放前固定，不按全年账单回选。\n\n主方案已完成{n}日的模拟结算总费为{total:,.4f}元，计划费{main_daily.planned_cost.sum():,.4f}元、紧急费{main_daily.emergency_cost.sum():,.4f}元、调整费0元。exp004同日期为{base_total:,.4f}元，差额{change:+,.4f}元（{100 * change / base_total:+.2f}%）。exp004使用不同控制器和爬坡协议，费用差为描述性对照，不能单独归因于周转惩罚。\n\n主方案{n * 144:,}个实际模拟执行区间中，超过1e−6 kWh的重叠有{observed_overlap}段；第二层全部规划节点中有{int(second_nodes.overlap_nodes.sum())}个重叠节点。连续LP没有互斥的数学保证，以下依据原始输出逐项检验。', ['period', 'daily', 'runs', 'overlap_summary']),
        ('题目指标及信息边界', '## 2. 题目指标及信息边界\n\n时间步长Δt=1/6小时；负载和光伏由kW乘Δt转为交流侧kWh。每天0:00锁定普通购电g，实际账单为Σp(g+5e)，e为实际紧急购电。普通计划全部计费，弃电w无收入，周转惩罚不是电费。\n\n单向效率η=√0.9；内部SOC保持1200—10800 kWh；单向功率不超过5000 kW，相邻十分钟净功率变化不超过1000 kW。SOC和末段净功率跨日传递，不设日末等于日初。D5-A允许紧急购电服务系统平衡，包括充电。\n\n每次滚动仅反馈当前十分钟平均供需，未来实际数据不进入规划。日内既不更改普通计划，也不更新场景权重。\n\n重叠原始量为min(c,d)，超过1e−6 kWh才计为重叠事件；数值容差内的原始残差仍保存。汇总重叠电量只累计超过阈值的事件。第一层参考解、第二层未来方案与实际执行分别统计，重复预测到的未来节点不能当作真实发生电量累加。四小时汇总段内先充后放，可能使两列都为正，不能据此判断十分钟重叠。', ['audit', 'overlap_summary']),
        ('数据与时间验证', '## 3. 数据与时间验证\n\n固定采用exp004/no_season_seed_42预测，来源分支提交b42168d5271097762953f38d472d0ef5fe1a908d；未重训或使用附件3、4。冻结档案SHA-256为6f7bcf27e420e2f05251820a5c1d8d1f87cd6234287e23505b33a51ae6d757b9。\n\n2月1日初始SOC=1421.7991105135516 kWh、上一段净功率=172.76 kW，承接相同一月预热。首日缺少历史预测误差供体，退化为一条点路径；之后只用已完整揭晓的历史日。评价区间为2—12月，不伪造1月新模型成绩。\n\n逐日独立核对供需平衡、SOC递推、单向边界、硬爬坡、跨日状态、普通计划锁定和五倍紧急费。两层节点按求解日志核对原约束和第二层预算。原模型、预测和执行相关13个源文件哈希未变。\n\n由于此前已查看过全年数据及实验成绩，本轮属于回顾性时序评价。严格遵守逐段可用信息，不把它称作未接触过的独立测试集。', ['audit', 'runs']),
        ('逐步技术讲解', f'## 4. 逐步技术讲解\n\n冻结午夜预测 → 抽取完整日联合误差 → 构造等权路径与前缀信息树 → 午夜两层LP → 锁定普通计划 → 当前反馈下逐段两层LP → 只执行当前动作 → 更新SOC与末段功率 → 按真实输入模拟结算。\n\n节点平衡为g+v+d+e=ℓ+c+w；内部能量递推为E后=E前+ηc−d/η；净充电功率P=(c−d)/Δt。节点概率q用于规划期望；同一信息节点共享动作，以避免使用未揭晓的场景身份。\n\n第一层最小化预期购电费用C。午夜包括普通计划和预期紧急费；日内普通计划已锁定，只优化剩余预期紧急费。第二层约束C≤(1+0.001)C*+0.0001元，目标为：\n\n`{penalty_formula}`\n\nT是本次剩余段数，Pmax=5000 kW，β无量纲。只改变β，不增设第三层。c+d既包括净充放电，也包括无意义循环，因为c+d=|c−d|+2min(c,d)。所以该项能抑制重叠，也会影响正常周转；不能把它写成仅对重叠收费或已标定的电池退化费。\n\n本轮完全移除0/1开关，模型为连续LP，不再使用MIP gap。仍按原连续LP精度求解，两层费用让步0.001与求解精度是不同概念。', ['preflight', 'runs']),
        ('实验设置', '## 5. 实验设置\n\n主方案β=0.1，对照β=0.01，均按用户确认的固定参数执行。最近最多56个完整历史日、最多16条整日负载/光伏联合路径、等权无放回抽样，种子42；06:00、12:00、18:00按已可见前缀二分，子节点至少2条路径。两组从同一初始状态开始，各自连续传递状态；两组只差周转权重。\n\nSciPy/HiGHS连续LP，每层60秒上限，原始/对偶可行性容差请求值1e−8，物理残差按1e−6验收。没有整数变量，也不以互斥作为停止条件；若出现重叠，保留原始数组和节点证据，继续松弛模型模拟。其他求解失败或原物理约束未通过仍停止。\n\nβ=0、0.001、0.01、0.1、1仅在20个已保存输入上作小范围诊断；该诊断不形成全年策略，也不按其费用替换预先确定的β=0.1。两组完整回放按顺序运行。CPU计时、回放墙钟、导出和报告时间分开记录，不把包含关系的计时相加。', ['runs', 'preflight', 'timing_history']),
        ('结果及失败案例', f'## 6. 结果及失败案例\n\n原LP在2月1日13:30求解窗口的未来14:10出现566.5937811694077 kWh重叠。同输入、同状态、同锁定计划的诊断中，β=0.01和0.1都将该窗口重叠降到数值阈值以下。下图同时展示充电与放电，表中保留总变差、周转量与费用，不能只看重叠下降。\n\n主方案当前实际执行重叠{observed_overlap}段；第一层参考解重叠{int(first_nodes.overlap_nodes.sum())}个节点，第二层方案重叠{int(second_nodes.overlap_nodes.sum())}个节点。第一层参考解用于提供费用基准，最终执行第二层的当前动作；第一层有重叠不等于实际电池同时充放电。\n\n已完成范围内最贵日期为{main_daily.loc[main_daily.total_cost.idxmax(), "date"]}，日费{main_daily.total_cost.max():,.4f}元。逐日费用、月度重叠分布、完整节点事件、实际充放电和SOC曲线均可检查。题目四个指定日期按完成状态列出；完整Excel在主方案334日完成并核验后生成。', ['preflight', 'preflight_curves', 'actual', 'daily', 'monthly', 'overlap_summary', 'overlap_timeline', 'overlap_heatmap', 'overlap_nodes', *specified]),
        ('历次指标和技术路线对比', '## 7. 历次指标和技术路线对比\n\nexp001—exp004原登记或重算成绩保留，标明评价期和可比性。exp004主要改变预测，执行器为确定性午夜MILP与日内贪心；本轮沿用它的同一冻结预测，改为有硬爬坡、信息树、两层LP与周转惩罚的逐段滚动策略。费用差是这些设计共同作用的结果，不能据此单独识别场景收益。\n\nβ=0.1与β=0.01只在共同完成日期上比较费用、净功率总变差和周转量。严格互斥旧版本的55日进度与旧LP失败证据独立归档，不混入本轮账单，也不将55日总额与334日总额排名。\n\n本轮没有新增预测模型。对冻结334日预测重算MAE、RMSE与WAPE，核对与exp004无季节组一致；光伏另列实际发电时段，分月数据与全时段指标分开。历史耗时保留设备、阶段、组数和统计范围，不把不同口径的时长作端到端速度排名。', ['paired_beta', 'history', 'forecast_history', 'forecast_breakdown', 'timing_history']),
        ('复现说明', '## 8. 复现说明\n\n原实验源码不变，新增入口位于`reports/experiments/exp005/soft_penalty/`。两组独立命令分别为：\n\n```sh\n.venv/bin/python reports/experiments/exp005/soft_penalty/run.py --beta 0.1 --out .work/reproduce-beta-0.1\n.venv/bin/python reports/experiments/exp005/soft_penalty/run.py --beta 0.01 --out .work/reproduce-beta-0.01\n```\n\n输出目录必须不存在。每个完整日保存场景、午夜方案、原始十分钟执行、两层求解日志和所有超过阈值的重叠节点；progress.json保存下一日初始状态。逐日诊断不净化充放电数组。\n\n重建报告：`.venv/bin/python reports/experiments/exp005/soft_penalty/build_report.py --complete --build`。完整证据表位于soft_penalty/evidence，费用与图表使用同一份原始数据。原连续LP与严格互斥报告分别归档于versions/lp-failure和versions/strict-paused。', ['audit', 'runs'])]
    cost_parts = (f'分项看，相比exp004，主方案普通计划费变化{main_daily.planned_cost.sum() - base_cost[0]:+,.4f}元，'
                  f'紧急购电费变化{main_daily.emergency_cost.sum() - base_cost[-1]:+,.4f}元。该分解解释账单差额，不构成对某一模型设计的因果归因。')
    sections[0] = (sections[0][0], sections[0][1] + '\n\n' + cost_parts, sections[0][2])
    sections[4] = (sections[4][0], sections[4][1].replace('CPU计时、回放墙钟、导出和报告时间分开记录，不把包含关系的计时相加。', '记录求解器累计耗时和回放墙钟，两者有包含关系，不相加。Excel导出与人工报告编制未独立计时，不补填估计实测值。'), sections[4][2])
    sections[5] = (sections[5][0], sections[5][1].replace('完整Excel在主方案334日完成并核验后生成。', workbook_text), sections[5][2])
    sections[5] = (sections[5][0], sections[5][1] + f'\n\nβ=0.01第二层未来规划共有{int(control_nodes.overlap_nodes.sum())}个超过阈值的重叠节点，最大{control_nodes.max_overlap_kwh.max():,.6f} kWh。以下残留案例沿包含最大重叠节点的一条祖先—后继路径展示规划充放电，属于当时的未来计划；它并未直接成为实际执行动作。', sections[5][2] + ['control_residual_case'])
    sections[6] = (sections[6][0], sections[6][1] + '\n\n' + beta_text + '\n\n本轮获批的完整回放仅含上述两个β。推导稿14.3建议的“点预测两层LP”及控制器分离对照尚未在本轮周转惩罚设置下完成；旧版点预测严格互斥进度保留，不能替代这一消融。因此本报告不宣称已经识别随机场景带来的独立收益。', sections[6][2] + ['timing_plot', 'technical_comparison'])
    sections[7] = (sections[7][0], sections[7][1] + '\n\n原模型11项测试已通过，覆盖单段五倍电价手算、共享节点后分支、满电硬爬坡的重叠反例、D5-A紧急充电、未来实际值扰动不改变之前动作、历史供体可见性与锁定普通计划。全年独立回放审计补充跨日跨月状态、实际结算与各层最优状态核对。\n\nExcel复现：先运行`soft_penalty/prepare_workbook.py`生成主方案表格载荷，再用未修改的`reports/export_workbooks_v2.mjs`通过artifact-tool导出，最后执行`soft_penalty/verify_workbook.py`回读核验。严格互斥旧版完整日另存于`data/results/exp005/paused-before-gap-review/saved-progress.zip`。', sections[7][2])
    definitions = {
        'actual': '每行一个已完成模拟执行十分钟区间；beta区分两条各自连续的策略。原始min(c,d)保留，事件阈值1e-6 kWh。',
        'overlap_summary': '午夜/日内和两层分别汇总。node是重复规划节点，不能当作真实发生区间或电量。',
        'overlap_nodes': '全部超过1e-6 kWh的规划节点事件；包含发布日期、回放时刻、未来时刻、层、概率和原始充放电。',
        'overlap_heatmap': '按月和时钟小时统计实际模拟执行重叠频率，分母是该月该小时已完成十分钟区间数，未观测不填零。',
        'preflight': '原失败窗口相同输入的第二层结果；不是全年费用或各组完整回放结果。',
        'control_residual_case': 'β=0.01第二层最大重叠事件，沿经过该节点的一条完整树路径截取前后3小时；未来规划，非实际执行。',
        'forecast_history': '历史原值，全部334日午夜预测，按原登记或新协议重算标签保留。',
        'forecast_breakdown': '重新核对同一冻结预测，光伏actual_generation仅取实际PV>0的区间。',
        'paired_beta': '两组共同完成日期，主方案β预先指定；没有以账单回选权重。',
        'history': '历史费用、协议及评价期，不同控制器仅描述性并列。',
        'timing_history': '阶段计时原值和范围；回放墙钟包含求解器，不相加。',
        'timing_plot': '历史训练、调度累计与本轮回放墙钟分别标注阶段；设备、组数和时间覆盖不同，只作描述性并列。',
        'technical_comparison': '对照各实验已有实现与本轮获批设置，描述预测、规划、执行、硬爬坡及互斥方式。'}
    for key, frame in frames.items():
        frame.to_csv(EVIDENCE / f'{key}.csv', index=False)
    snapshot = read(REPORT / 'app/src/data.json')
    snapshot.update(title=title, variant='soft-penalty', buildStatus='complete' if args.complete else 'updating',
        generatedAt=datetime.now(UTC).isoformat(), report={'asOf': last},
        sections=[{'title': title, 'markdown': text, 'queryIds': ids} for title, text, ids in sections],
        queries={key: {'rows': rows(frame), 'source': {'label': 'exp005周转惩罚实测与核验',
            'files': [f'soft_penalty/evidence/{key}.csv'],
            'metricDefinitions': [{'label': key, 'definition': definitions.get(key, '已完成日期原始数组及求解记录的独立核对和汇总；未完成日期不填零。')}],
            'evidenceFlow': [{'title': '保存原始输出并核对', 'detail': 'soft_penalty/evidence.py核对原始约束、账单及信息输入；原实验源码不变。'}]}}
            for key, frame in frames.items()})
    dump(REPORT / 'app/src/data.json', snapshot)
    dump(REPORT / 'reviewed.json', snapshot)
    record = {'experiment': 'exp005', 'variant': 'soft-penalty', 'main_beta': .1, 'control_beta': .01,
              'complete': finished, 'runs': rows(runs), 'period': period_rows, 'paired_beta': common_rows,
              'overlap_summary': rows(frames['overlap_summary']), 'original_code_unchanged': True,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()},
              'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(EVIDENCE.glob('*.csv'))}}
    dump(REPORT / 'soft-penalty-record.json', record)
    figure_dir = Path(__file__).parent / 'figures'
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
    chart_period = [r for r in period_rows if r['beta'] in [None, .1]]
    ax.barh([r['label'] for r in chart_period], [r['total_wan'] for r in chart_period], color=['#8c9298', '#287f8e'])
    ax.set_xlabel(f'模拟结算费用 / 万元（共同{n}日）')
    for i, r in enumerate(chart_period):
        ax.text(r['total_wan'], i, f' {r["total_wan"]:.2f}', va='center')
    for ext in ['png', 'svg']:
        fig.savefig(figure_dir / f'cost-comparison.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, sharey=True, layout='constrained')
    for ax, beta in zip(axes, [0., .01, .1]):
        sub = frames['preflight_curves'].query('beta == @beta and hour <= 16.5')
        ax.plot(sub.hour, sub.charge_kwh, label='充电', color='#287f8e')
        ax.plot(sub.hour, sub.discharge_kwh, label='放电', color='#b75c3c', linestyle='--')
        ax.set_title(f'相同原失败输入：β={beta:g}')
        ax.set_ylabel('kWh / 十分钟')
    axes[0].legend()
    axes[-1].set_xlabel('时刻 / 小时')
    for ext in ['png', 'svg']:
        fig.savefig(figure_dir / f'overlap-same-input.{ext}', dpi=180)
    plt.close(fig)
    from report_figures import render_additional_figures, markdown_evidence
    render_additional_figures(frames, figure_dir)
    text = [f'# {title}\n\n{progress_text}']
    for i, (_, content, _) in enumerate(sections, 1):
        if i == 1:
            content += '\n\n![同日期费用](soft_penalty/figures/cost-comparison.png)'
        if i == 6:
            content += '\n\n![原失败输入的惩罚对照](soft_penalty/figures/overlap-same-input.png)'
            content += '\n\n![两层规划重叠](soft_penalty/figures/planning-overlap.png)\n\n![实际执行月时分布](soft_penalty/figures/actual-overlap-heatmap.png)'
            if len(frames['control_residual_case']):
                content += '\n\n![较小惩罚的残留规划重叠](soft_penalty/figures/control-residual-overlap.png)'
        if i == 7:
            content += '\n\n![历史费用](soft_penalty/figures/history-cost.png)\n\n![历史预测误差](soft_penalty/figures/history-forecast.png)\n\n![历史阶段耗时](soft_penalty/figures/history-timing.png)'
        content += markdown_evidence(i, frames)
        (REPORT / f'section-{i}.md').write_text(content + '\n')
        text.append(content)
    (REPORT / 'report.md').write_text('\n\n'.join(text) + '\n')
    (REPORT / 'README.md').write_text(f'# {title}\n\n{progress_text}\n\n主方案β=0.1，对照β=0.01，原实验源码未修改。报告见report.html/report.md。原结果保存在versions，新增运行和核验代码在soft_penalty。\n')
    if workbook_ready:
        shutil.copy2(ROOT / 'data/results/exp005/result2.xlsx', REPORT / 'result2.xlsx')
    if args.build:
        node = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
        plugin = Path.home() / '.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2/scripts/data-app.mjs'
        subprocess.run([str(node), str(plugin), 'build', '--project-dir', str(REPORT / 'app'), '--separate-data'], check=True)
        offline = REPORT / 'app/.data-app-offline/exports/report.html'
        subprocess.run([str(node), str(plugin), 'export-offline', '--project-dir', str(REPORT / 'app'), '--output', str(offline)], check=True)
        shutil.copy2(offline, REPORT / 'report.html')
        shutil.copytree(EVIDENCE, REPORT / 'app/dist/soft_penalty/evidence', dirs_exist_ok=True)
        if workbook_ready:
            shutil.copy2(REPORT / 'result2.xlsx', REPORT / 'app/dist/result2.xlsx')
    print(json.dumps({'days': n, 'finished': finished, 'actual_overlap_intervals': observed_overlap,
                      'total_cost': total, 'difference_vs_exp004': change}, ensure_ascii=False))


if __name__ == '__main__':
    main()
