"""Render conversation comparisons from final reviewed evidence, with source receipts."""
import json
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).parent / 'evidence'
OUT = Path('/Users/vegetarianwolf/.codex/visualizations/2026/09/12/01a09588-1bd9-7082-9e2b-c30d8191bd77')
NODE = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
PLUGIN = Path.home() / '.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2/skills/visualize-data'


def write(name, value):
    path = OUT / f'{name}.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return path


def render(name, frame, title, y, unit, source):
    data = json.loads(frame.to_json(orient='records', double_precision=15))
    columns = list(frame.columns)
    payload = {'schemaVersion': 1, 'id': name, 'queryId': name, 'title': title, 'height': 360,
        'chart': {'type': 'horizontalBar', 'x': 'label', 'y': y, 'xLabel': unit, 'valueDecimals': 4, 'stackable': False},
        'rows': data, 'columns': columns, 'source': source}
    path = write(name, payload)
    subprocess.run([str(NODE), str(PLUGIN / 'scripts/render-inline-chart.mjs'), '--input', str(path), '--output', str(OUT / f'{name}.html')], check=True)
    receipt = {'schemaVersion': 1, 'items': [{'id': name, 'title': title, 'queries': [{'id': name,
        'source': source, 'rows': data, 'columns': columns, 'reportingPeriod': '2025-02-01—2025-12-31，334日'}]}]}
    path = write(name + '-sources', receipt)
    subprocess.run([str(NODE), str(PLUGIN / 'scripts/render-inline-sources.mjs'), '--input', str(path), '--output', str(OUT / f'{name}-sources.html')], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(EVIDENCE / 'runs.csv')
    assert len(runs) == 2 and all(runs.status == 'complete') and all(runs.completed_days == 334)
    history = pd.read_csv(EVIDENCE / 'history.csv')
    selection = ['exp001 同口径重算', 'exp002 正式风险', 'exp003 正式', 'exp004 无季节']
    cost = history[history.label.isin(selection) | history.label.str.startswith('exp005 周转惩罚')][['label', 'total_cost']].copy()
    cost['total_wan'] = cost.total_cost / 10000
    cost_source = {'label': '问题二历次费用与本轮回放', 'files': ['reports/experiments/exp005/soft_penalty/evidence/history.csv'],
        'metricDefinitions': [{'label': '模拟结算费', 'definition': '模拟结算费是普通计划费与五倍电价的实际紧急购电费之和；图中单位为万元。'}],
        'filters': ['正式主种子42', 'exp001使用同口径重算值', 'exp004选无季节组'],
        'caveats': ['exp005新增硬爬坡与场景滚动控制，和此前控制器仅作描述性并列；费用差不能单独归因于周转惩罚。'],
        'evidenceFlow': [{'kind': 'validation', 'title': '账单核对', 'detail': '本轮两组原始实际轨迹逐段重算费用，与每日保存账单之差小于1e−6元。'}]}
    render('q2-history-cost', cost, '历次334日费用：不同控制器的描述性对照', 'total_wan', '模拟结算费 / 万元', cost_source)
    forecasts = pd.read_csv(EVIDENCE / 'forecast_history.csv')
    labels = ['exp001 午夜重评分', 'exp002 午夜重评分', 'exp003 正式', 'exp004 无季节', 'exp005 周转惩罚·同一冻结预测']
    forecast = forecasts[(forecasts.target == 'net_load') & forecasts.label.isin(labels)].copy()
    for metric, unit, name in [('wape_pct', 'WAPE / %', 'wape'), ('rmse', 'RMSE / kW', 'rmse')]:
        source = {'label': '问题二午夜净负荷预测误差', 'files': ['reports/experiments/exp005/soft_penalty/evidence/forecast_history.csv'],
            'metricDefinitions': [{'label': unit, 'definition': '净负荷为负载减光伏；指标覆盖334日全部48,096个午夜预测十分钟点。'}],
            'filters': ['目标：净负荷', '全时段', '主种子42', 'exp001与exp002使用午夜重评分'],
            'caveats': ['exp005直接沿用exp004无季节冻结预测，未重新训练，预测误差完全相同。负载、光伏及实际发电时段指标另见完整报告。'],
            'evidenceFlow': [{'kind': 'validation', 'title': '冻结预测核对', 'detail': '重新计算冻结预测的MAE、RMSE和WAPE，与exp004记录绝对差均小于1e−8。'}]}
        render('q2-net-load-' + name, forecast[['label', metric]], '午夜净负荷预测：' + unit, metric, unit, source)
    timing = pd.read_csv(EVIDENCE / 'timing_plot.csv')
    chosen = ['exp001 同口径重算', 'exp002 正式风险', 'exp003 正式', 'exp004 无季节', 'β=0.1', 'β=0.01']
    timing = timing[timing.label.isin(chosen) & timing.stage.isin(['调度执行', '回放墙钟'])][['item', 'seconds', 'scope']].rename(columns={'item': 'label'})
    timing_source = {'label': '问题二历史阶段计时', 'files': ['reports/experiments/exp005/soft_penalty/evidence/timing_history.csv'],
        'metricDefinitions': [{'label': '阶段耗时', 'definition': '历史值为主种子334日调度执行累计，本轮为各自334日连续回放墙钟，均以秒显示。'}],
        'filters': ['主种子42', '历史正式或指定重算控制器'],
        'caveats': ['计时阶段、设备和实现不同，只作描述性比较，不能解释为端到端速度变化；本轮没有重新训练预测模型。']}
    render('q2-history-timing', timing, '334日计算代价：累计调度与回放墙钟分列', 'seconds', '秒', timing_source)
    diagnostics = pd.read_csv(EVIDENCE / 'period.csv').dropna(subset=['beta'])[['label', 'days', 'overlap_intervals']]
    data = json.loads(diagnostics.to_json(orient='records'))
    source = {'label': '连续回放执行重叠核验', 'files': ['reports/experiments/exp005/soft_penalty/evidence/actual.csv', 'reports/experiments/exp005/soft_penalty/evidence/overlap_summary.csv'],
        'metricDefinitions': [{'label': '实际重叠', 'definition': '每个十分钟内min(充电量,放电量)>1e−6 kWh才计作实际执行重叠。'}],
        'caveats': ['第一层参考解可能重叠，实际执行第二层当前动作；连续LP没有严格互斥保证。'],
        'evidenceFlow': [{'kind': 'validation', 'title': '原始数组核对', 'detail': '逐日检查原始充放电数组，保留数值残差，不进行互斥投影或净化。'}]}
    receipt = {'schemaVersion': 1, 'items': [{'id': 'q2-executed-overlap', 'title': '两组实际执行的充放电重叠', 'queries': [{'id': 'executed-overlap', 'rows': data, 'columns': list(diagnostics.columns), 'source': source, 'reportingPeriod': '2025-02-01—2025-12-31，334日'}]}]}
    path = write('q2-overlap-sources', receipt)
    subprocess.run([str(NODE), str(PLUGIN / 'scripts/render-inline-sources.mjs'), '--input', str(path), '--output', str(OUT / 'q2-overlap-sources.html')], check=True)


if __name__ == '__main__':
    main()
