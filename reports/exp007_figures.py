"""Standalone PNG/SVG exports using the same frozen tables as the report."""

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cumcm-exp007-mpl')
import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from reports.exp007_evidence import CORE, REPORT, RUNS, sha256

COLORS = {'exp004/no_season': '#5f777b', 'exp006/primary': '#8b7398',
          'exp006/greedy_execution': '#367ba0', 'exp007/regularized_42': '#ad6426',
          'exp007/cost_only_42': '#ba7580'}
SHORT = {'exp004/no_season': 'exp004 无季节基线', 'exp006/primary': 'exp006 正式·树DP',
         'exp006/greedy_execution': 'exp006 贪心执行对照',
         **{'exp007/'+r: 'exp007 '+('轻惩罚' if r.startswith('regularized') else '仅费用')+'·'+r.rsplit('_', 1)[1] for r in RUNS}}
STYLES = ['-', '--', '-.', '-', ':']


def build_figures(frames, battery):
    for path in (Path('/System/Library/Fonts/PingFang.ttc'), Path('/System/Library/Fonts/STHeiti Medium.ttc')):
        if path.is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'path.simplify': False, 'agg.path.chunksize': 10000,
                         'axes.edgecolor': '#777777', 'grid.color': '#dddddd',
                         'grid.alpha': .55, 'figure.facecolor': '#ffffff',
                         'savefig.facecolor': '#ffffff'})
    directory = REPORT/'figures'
    directory.mkdir(exist_ok=True)
    outputs = []
    def save(fig, name, title, source_tables, points=None):
        fig.canvas.draw()
        for extension in ('png', 'svg'):
            path = directory/f'{name}.{extension}'
            fig.savefig(path, dpi=170, bbox_inches='tight', pad_inches=.16)
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        outputs.append({'id': name, 'title': title, 'png': f'figures/{name}.png',
                        'svg': f'figures/{name}.svg', 'source_tables': source_tables,
                        'png_sha256': sha256(directory/f'{name}.png'),
                        'svg_sha256': sha256(directory/f'{name}.svg'),
                        'raw_points_per_series': points, 'smoothing': False,
                        'rendered': True})
        plt.close(fig)

    def color(policy):
        if policy in COLORS:
            return COLORS[policy]
        if policy.startswith('exp007/regularized'):
            return '#ad6426'
        if policy.startswith('exp007/cost_only'):
            return '#ba7580'
        return '#8a9394'

    def hbars(ax, rows, key, unit, scale=1, labels=None):
        values = rows[key].to_numpy(float)/scale
        y = np.arange(len(rows))
        names = labels if labels is not None else [SHORT.get(r.policy_id, r.label) for r in rows.itertuples()]
        ax.barh(y, values, color=[color(r.policy_id) for r in rows.itertuples()], height=.62)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        upper = max(float(np.max(values)), 1.)
        ax.set_xlim(0, upper*1.23)
        for i, value in enumerate(values):
            ax.text(value+upper*.014, i, f'{value:,.2f}', va='center', fontsize=10)

    costs = frames['cost_history']
    historic_ids = ['exp001/legacy_rebased', 'exp002/primary', 'exp003/primary',
                    'exp004/no_season', 'exp004/causal_season', 'exp006/primary',
                    'exp006/greedy_execution', 'exp007/regularized_42', 'exp007/cost_only_42']
    p = costs.set_index('policy_id').loc[historic_ids].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.3), layout='constrained',
                             gridspec_kw={'width_ratios': [1.15, 1]})
    hbars(axes[0], p, 'total_cost', '总购电费用 / 万元', 10000)
    axes[0].set_title('同物理结算口径的历史费用', loc='left')
    y = np.arange(len(p))
    axes[1].barh(y, p.planned_cost/10000, label='计划购电费', color='#6a9299', height=.62)
    axes[1].barh(y, p.emergency_cost/10000, left=p.planned_cost/10000,
                 label='5倍紧急购电费', color='#b27a46', height=.62)
    axes[1].set_yticks(y, [SHORT.get(r.policy_id, r.label) for r in p.itertuples()])
    axes[1].invert_yaxis(); axes[1].grid(axis='x'); axes[1].set_axisbelow(True)
    axes[1].set_xlabel('分项购电费用 / 万元')
    fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    axes[1].set_title('总费用的计划费与紧急费构成', loc='left')
    fig.suptitle('2025-02-01—12-31 · 334日；保留实验角色，不按测试费用更换主组', fontsize=14)
    save(fig, 'history-costs', '历次费用与分项', ['cost_history'])

    descriptive = costs[(costs.policy_id == 'exp001/original') | (costs.experiment == 'exp005') | (costs.policy_id == 'exp004/oracle_season')]
    fig, ax = plt.subplots(figsize=(12, 4.4), layout='constrained')
    hbars(ax, descriptive, 'total_cost', '已登记/核验总费用 / 万元（不排名、不计算改善率）', 10000)
    ax.set_title('不同物理、信息或执行协议的历史记录 · 仅描述性并列', loc='left')
    save(fig, 'history-descriptive-only', '不可直接排名的历史记录', ['cost_history'])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout='constrained')
    monthly = frames['monthly_cost']
    for j, policy in enumerate(CORE):
        group = monthly[monthly.policy_id == policy].sort_values('month')
        for ax, key, title in zip(axes, ('total_cost', 'emergency_cost'),
                                  ('月度总费用', '月度紧急购电费'), strict=True):
            ax.plot(group.month, group[key]/10000, color=color(policy), linestyle=STYLES[j],
                    marker='o', markersize=4, label=SHORT[policy])
            ax.set_title(title, loc='left'); ax.set_ylabel('万元'); ax.grid(); ax.set_xticks(range(2, 13))
    axes[0].legend(ncol=3, loc='upper left', fontsize=10)
    axes[-1].set_xlabel('2025年月份')
    save(fig, 'monthly-costs', '同预测方案的月度费用', ['monthly_cost'])

    runs = frames['all_runs'].set_index('run_id').loc[RUNS].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    hbars(axes[0], runs, 'total_cost', '334日总费用 / 万元', 10000)
    axes[0].set_title('六个预声明策略训练任务', loc='left')
    seeds = [42, 2026, 3407]
    for family, col, offset in [('regularized', '#ad6426', -.13), ('cost_only', '#ba7580', .13)]:
        group = runs[runs.family == family].set_index('rl_seed').loc[seeds]
        axes[1].scatter(np.arange(3)+offset, group.total_cost/10000, color=col, s=80,
                        label='轻惩罚' if family == 'regularized' else '仅费用')
        for i, value in enumerate(group.total_cost/10000):
            axes[1].annotate(f'{value:.1f}', (i+offset, value), xytext=(0, 8 if family == 'regularized' else -15),
                             textcoords='offset points', ha='center', fontsize=10)
    axes[1].set_xticks(range(3), [f'RL种子 {s}' for s in seeds])
    axes[1].set_ylabel('总费用 / 万元（局部纵轴）'); axes[1].grid(axis='y'); axes[1].legend()
    axes[1].margins(y=.22); axes[1].set_title('相同预测 seed42，独立 RL 训练种子', loc='left')
    save(fig, 'seed-costs', '六组费用与随机种子离散程度', ['all_runs', 'seed_summary'])

    metrics = frames['battery_history'].set_index('policy_id').loc[CORE].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout='constrained')
    for ax, key, title, unit, scale in zip(axes.flat,
            ['throughput_kwh', 'equivalent_full_cycles', 'nonidle_direction_reversals', 'total_variation_kw'],
            ['交流侧充放电吞吐', '电芯侧等效全循环', '跳过空闲槽的方向反转', '全评价净功率总变差'],
            ['万 kWh', 'EFC（运行强度代理）', '次数', '百万 kW/相邻10分钟槽'],
            [10000, 1, 1, 1e6], strict=True):
        hbars(ax, metrics, key, unit, scale); ax.set_title(title, loc='left')
    save(fig, 'battery-comparison', '电池周转与变化指标', ['battery_history'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout='constrained')
    for ax, key, title, unit in zip(axes.flat,
            ['mean_absolute_change_kw', 'rms_change_kw', 'direct_adjacent_reversals', 'large_jump_pct'],
            ['相邻槽平均绝对功率变化', '相邻槽功率变化 RMS', '相邻两槽直接充放反转', '|ΔP| > 1000 kW 的相邻槽比例'],
            ['kW/相邻10分钟槽', 'kW/相邻10分钟槽', '次数；空闲不跨越计数', '%；描述阈值，非硬约束'], strict=True):
        hbars(ax, metrics, key, unit); ax.set_title(title, loc='left')
    save(fig, 'power-variation', '完整功率波动诊断', ['power_variation'])

    forecast = frames['forecast_history']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for row, target in enumerate(('load', 'pv')):
        group = forecast[(forecast.target == target) & (forecast.population == 'all') & ~forecast.exploratory]
        group = group[(~group.experiment.isin(['exp005', 'exp006', 'exp007']))
                      | group.policy_id.isin(['exp006/primary', 'exp007/regularized_42'])].copy()
        for column, metric in enumerate(('rmse', 'wape_pct')):
            ax = axes[row, column]
            hbars(ax, group, metric, 'kW' if metric == 'rmse' else '%')
            ax.set_title(('负载' if target == 'load' else '光伏')+(' RMSE' if metric == 'rmse' else ' WAPE'), loc='left')
    fig.suptitle('同48,096个午夜预测槽 · exp007完整复用 exp004 无季节预测，没有预测精度改进', fontsize=13)
    save(fig, 'history-errors', '历次预测误差', ['forecast_history'])

    timing = frames['timings']
    p = timing[(timing.experiment == 'exp007') & (timing.name == 'regularized_42')
               & (timing.stage != '本次组墙钟')].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    y = np.arange(len(p)); axes[0].barh(y, p.seconds, color='#ad6426'); axes[0].set_yticks(y, p.stage)
    axes[0].invert_yaxis(); axes[0].set_xlabel('累计实测秒'); axes[0].grid(axis='x')
    upper = max(float(p.seconds.max()), 1)
    axes[0].set_xlim(0, upper*1.2)
    for i, value in enumerate(p.seconds):
        axes[0].text(value+upper*.014, i, f'{value:.3f}', va='center', fontsize=10)
    axes[0].set_title('exp007正式主组的计算阶段', loc='left')
    hist = timing[(timing.stage.isin(['训练', 'RL策略训练']))
                  & (~timing.experiment.isin(['exp006', 'exp007']) | (timing.name == 'regularized_42'))].copy()
    y = np.arange(len(hist)); axes[1].barh(y, hist.seconds, color='#8a9394'); axes[1].set_yticks(y, hist.label)
    axes[1].invert_yaxis(); axes[1].set_xlabel('实测秒；任务规模/设备不同，仅描述'); axes[1].grid(axis='x')
    upper = max(float(hist.seconds.max()), 1)
    axes[1].set_xlim(0, upper*1.2)
    for i, value in enumerate(hist.seconds):
        axes[1].text(value+upper*.014, i, f'{value:.2f}', va='center', fontsize=10)
    axes[1].set_title('历史预测训练与本次策略训练', loc='left')
    save(fig, 'stage-timings', '阶段计时及历史训练范围', ['timings'])

    t = frames['training']
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
    for run in ('regularized_42', 'cost_only_42'):
        group = t[t.run_id == run]
        x = group.cumulative_transitions/1e6
        for ax, key, title in zip(axes.flat,
                ['mean_episode_cost_yuan', 'entropy', 'value_loss', 'approx_kl'],
                ['历史训练日平均执行费用', '策略熵', '价值网络损失', 'PPO近似 KL'], strict=True):
            if key in group:
                ax.plot(x, group[key], color=color('exp007/'+run), linewidth=.8,
                        alpha=.85, label=SHORT['exp007/'+run])
            ax.set_title(title, loc='left'); ax.set_xlabel('累计训练状态转移 / 百万'); ax.grid()
    axes[0, 0].set_ylabel('元 / 模拟历史日'); axes[0, 0].legend(fontsize=10)
    fig.suptitle('原始训练诊断（未平滑）· 不同时点训练历史会变化，不能作为测试收益曲线', fontsize=13)
    save(fig, 'learning-curves', '训练学习曲线及优化诊断', ['training'])

    paired = frames['paired_daily'].copy()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    x = pd.to_datetime(paired.date)
    axes[0].plot(x, paired.difference_yuan/1000, color='#ad6426', linewidth=.9)
    axes[0].axhline(0, color='#555555', linewidth=.8); axes[0].set_ylabel('本次－基线 / 千元')
    axes[0].set_title('每日费用差：exp007正式主组 对 exp004无季节', loc='left')
    axes[1].plot(x, paired.difference_yuan.cumsum()/10000, color='#ad6426', linewidth=1.4)
    axes[1].axhline(0, color='#555555', linewidth=.8); axes[1].set_ylabel('累计费用差 / 万元')
    axes[1].set_title('累计费用差（正值表示本次更贵）', loc='left')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月')); ax.grid()
    save(fig, 'daily-cost-differences', '同日费用与累计差异', ['paired_daily'])

    worst = paired.loc[paired.difference_yuan.idxmax(), 'date']
    detail = frames['detail']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), layout='constrained')
    for policy in ['exp004/no_season', 'exp007/regularized_42']:
        group = detail[(detail.date == worst) & (detail.policy_id == policy)]
        axes[0].plot(group.hour, group.planned_kwh, label=SHORT[policy]+' 计划购电', color=color(policy))
        axes[1].plot(group.hour, group.emergency_kwh, label=SHORT[policy], color=color(policy))
        axes[2].plot(group.hour, group.soc_end_kwh, label=SHORT[policy], color=color(policy))
    group = detail[(detail.date == worst) & (detail.policy_id == 'exp007/regularized_42')]
    axes[0].plot(group.hour, group.actual_net_kwh, color='#555555', linestyle=':', label='实际净负荷')
    for ax, label in zip(axes, ['净负荷/计划购电 kWh', '紧急购电 kWh', 'SOC kWh'], strict=True):
        ax.set_ylabel(label); ax.legend(ncol=3, fontsize=10); ax.grid(); ax.set_xticks(range(0, 25, 4)); ax.set_xlim(0, 24)
    axes[-1].set_xlabel('区间终点小时（00:10 对应 00:00—00:10）')
    fig.suptitle(f'负面案例：相对 exp004 费用增加最多的一天 · {worst}', fontsize=14)
    save(fig, 'failure-case', '费用差最不利日的供需、应急与SOC', ['detail', 'paired_daily'], 144)

    by_policy = {row['policy_id']: row for row in battery['full_curves']}
    sampled = {(row['policy_id'], row['date']): row for row in battery['random_curves']}
    def power_axes(ax):
        ax.axhline(0, color='#555555', linewidth=.7)
        ax.axhline(5000, color='#929292', linestyle=':', linewidth=.8)
        ax.axhline(-5000, color='#929292', linestyle=':', linewidth=.8)
        ax.set_ylim(-5500, 5500); ax.set_yticks([-5000, 0, 5000]); ax.grid(axis='x')
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout='constrained')
    for ax, date in zip(axes.flat, battery['dates'], strict=True):
        for j, policy in enumerate(CORE):
            row = sampled[policy, date]
            ax.plot(row['hours_end'], row['power_kw'], label=SHORT[policy],
                    color=color(policy), linestyle=STYLES[j], linewidth=1.05)
        power_axes(ax); ax.set_title(date, loc='left'); ax.set_xlabel('区间终点小时'); ax.set_ylabel('净电池功率 / kW')
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=3, fontsize=10)
    fig.suptitle('固定随机四日 · 充电为正，放电为负 · 每日144个原始点，无平滑', fontsize=14)
    save(fig, 'battery-random-days', '同日期随机四日实际净电池功率', ['battery_random_days'], 144)

    timestamps = pd.date_range('2025-02-01 00:10', periods=48096, freq='10min')
    fig, axes = plt.subplots(len(battery['policies']), 1, figsize=(16, 2.05*len(battery['policies'])),
                             layout='constrained', sharex=True)
    for ax, policy in zip(axes, battery['policies'], strict=True):
        row = by_policy[policy]
        ax.plot(timestamps, row['net_power_kw'], color=color(policy), linewidth=.35)
        power_axes(ax); ax.set_ylabel('kW'); ax.set_title(SHORT[policy], loc='left', fontsize=11)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    axes[-1].set_xlim(timestamps[0], timestamps[-1])
    axes[-1].set_xlabel('2025年正式评价；最末01月刻度为2026-01-01 00:00（12月31日最后区间终点）')
    fig.suptitle('334日实际电池净功率 · 每方案完整48,096点 · 不含1月预热 · 统一纵轴，无平滑', fontsize=14)
    save(fig, 'battery-full-evaluation', '九个方案的完整334日净电池功率', ['battery/full_curves.npz'], 48096)
    # Each policy also gets its own standalone, full-resolution publication files.
    for policy in battery['policies']:
        name = policy.replace('/', '__')
        row = by_policy[policy]
        fig, ax = plt.subplots(figsize=(16, 4.5), layout='constrained')
        ax.plot(timestamps, row['net_power_kw'], color=color(policy), linewidth=.4)
        power_axes(ax); ax.set_ylabel('净电池功率 / kW'); ax.set_title(SHORT[policy]+' · 334日完整48,096点，无平滑', loc='left')
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.set_xlabel('2025年正式评价；最末01月刻度为2026-01-01 00:00（12月31日最后区间终点）')
        save(fig, name+'-full', SHORT[policy]+' 完整评价功率', [row['csv']], 48096)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), layout='constrained')
        for ax, date in zip(axes.flat, battery['dates'], strict=True):
            curve = sampled[policy, date]
            ax.plot(curve['hours_end'], curve['power_kw'], color=color(policy), linewidth=1.1)
            power_axes(ax); ax.set_title(date, loc='left'); ax.set_ylabel('kW'); ax.set_xlabel('区间终点小时')
            ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
        fig.suptitle(SHORT[policy]+' · 固定随机四日 · 144点/日，无平滑', fontsize=14)
        save(fig, name+'-random', SHORT[policy]+' 随机四日功率', ['battery_random_days'], 144)
    return outputs


if __name__ == '__main__':
    from reports.exp007_evidence import read
    payload = read(REPORT/'evidence/report_data.json')
    print(len(build_figures({key: pd.DataFrame(rows) for key, rows in payload['tables'].items()}, payload['battery'])))
