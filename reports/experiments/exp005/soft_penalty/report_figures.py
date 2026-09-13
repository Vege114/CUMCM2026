"""Publication figures and Markdown tables from the same reviewed report frames."""

import numpy as np
import matplotlib.pyplot as plt


def save(fig, directory, name):
    for ext in ['png', 'svg']:
        fig.savefig(directory / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def render_additional_figures(frames, directory):
    sub = frames['control_residual_case']
    if len(sub):
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        ax.plot(sub.hour, sub.charge_kwh, label='规划充电', color='#287f8e')
        ax.plot(sub.hour, sub.discharge_kwh, label='规划放电', color='#b75c3c', linestyle='--')
        events = sub[sub.overlap_kwh > 1e-6]
        ax.scatter(events.hour, events.overlap_kwh, label='min(c,d)>1e−6', color='#795899', zorder=4)
        ax.set_title(f'β=0.01 · {sub.date.iloc[0]} 残留第二层重叠（未来规划，非执行）')
        ax.set_xlabel('时刻 / 小时')
        ax.set_ylabel('kWh / 十分钟')
        ax.legend()
        save(fig, directory, 'control-residual-overlap')
    history = frames['history']
    selected = ['exp001 同口径重算', 'exp002 正式风险', 'exp003 正式', 'exp004 无季节']
    sub = history[history.label.isin(selected) | (history.label.str.startswith('exp005 周转惩罚') & history.period.str.contains('334天'))]
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.barh(sub.label, sub.total_cost / 10000, color=['#8b979a' if not x.startswith('exp005') else '#287f8e' for x in sub.label])
    ax.invert_yaxis()
    ax.set_xlabel('2—12月模拟结算费用 / 万元；控制器不同，描述性并列')
    for i, value in enumerate(sub.total_cost / 10000):
        ax.text(value, i, f' {value:.2f}', va='center')
    save(fig, directory, 'history-cost')
    labels = ['exp001 午夜重评分', 'exp002 午夜重评分', 'exp003 正式', 'exp004 无季节', 'exp005 周转惩罚·同一冻结预测']
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for i, target in enumerate(['load', 'pv', 'net_load']):
        sub = frames['forecast_history'].query('target == @target')
        sub = sub[sub.label.isin(labels)].set_index('label').reindex(labels)
        for j, (metric, unit) in enumerate([('wape_pct', 'WAPE / %'), ('rmse', 'RMSE / kW')]):
            ax = axes[i, j]
            ax.barh(labels, sub[metric], color='#287f8e')
            ax.invert_yaxis()
            ax.set_title(target)
            ax.set_xlabel(unit)
    fig.suptitle('334日午夜预测；exp005直接沿用exp004冻结预测')
    save(fig, directory, 'history-forecast')
    sub = frames['timing_plot']
    fig, ax = plt.subplots(figsize=(11, 8), layout='constrained')
    ax.barh(sub.item, sub.seconds, color='#687a96')
    ax.invert_yaxis()
    ax.set_xlabel('秒；阶段、设备与组数不同，不能作端到端速度排名')
    save(fig, directory, 'history-timing')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout='constrained')
    for ax, beta in zip(axes, [.1, .01]):
        sub = frames['overlap_timeline'].query('beta == @beta')
        for layer, color in [(1, '#b75c3c'), (2, '#287f8e')]:
            group = sub.query('layer == @layer')
            ax.plot(np.arange(len(group)), group.max_overlap_kwh, label=f'第{layer}层', color=color)
        ax.set_title(f'β={beta:g}；规划节点包含重复预测，非实际发生电量')
        ax.set_ylabel('最大重叠 / kWh')
        ax.legend()
    axes[-1].set_xlabel('已完成回放日序号（0=2月1日）')
    save(fig, directory, 'planning-overlap')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
    vmax = max(.01, frames['overlap_heatmap'].overlapRate.max())
    for ax, beta in zip(axes, [.1, .01]):
        sub = frames['overlap_heatmap'].query('beta == @beta')
        matrix = sub.pivot(index='month', columns='hour_band', values='overlapRate')
        im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=vmax, cmap='Blues')
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xticks(range(0, 24, 2), [f'{h:02}:00' for h in range(0, 24, 2)])
        ax.set_title(f'β={beta:g} 实际执行重叠频率；事件阈值1e−6 kWh')
        fig.colorbar(im, ax=ax, label='重叠区间数 / 该月该小时区间数')
    save(fig, directory, 'actual-overlap-heatmap')


def markdown_evidence(section, frames):
    chosen = {1: ['period'], 5: ['runs'], 6: ['preflight', 'overlap_summary', 'overlap_nodes', 'monthly', 'specified_plan', 'specified_battery', 'specified_soc', 'specified_emergency'],
              7: ['paired_beta', 'history', 'forecast_history', 'timing_history', 'technical_comparison'], 8: ['audit']}.get(section, [])
    blocks = []
    for key in chosen:
        frame = frames[key]
        if frame.empty:
            continue
        def cell(value):
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    return '—'
                return f'{value:.6g}' if value != 0 and abs(value) < 1e-5 else f'{value:,.6f}'.rstrip('0').rstrip('.')
            return str(value).replace('|', '\\|').replace('\n', ' ')
        blocks.append('\n\n' + key + '（原值可查对应CSV）\n\n| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)))
    return ''.join(blocks)
