"""Reproducible runtime estimate; approximate full replay remains unstarted."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'data/results/exp005/gap-timing-review'
design = json.loads((OUT / 'design.json').read_text())
results = json.loads((OUT / 'results.json').read_text())
stress = json.loads((OUT / 'stress.json').read_text())
windows = json.loads((ROOT / '.work/exp005/runtime_windows.json').read_text())
assert len(results) == 54 and all(r['success'] for r in results)


def stratum(window):
    return ('midnight' if window['slot'] == -1 else
            list(design['strata'])[1 + int(np.searchsorted([.2, .75, 2., 7.], window['seconds'], side='right'))])


rows = []
for gap, interval in [(0., [5, 7]), (.0005, [4.5, 6.5]), (.001, [4, 6])]:
    ratios = {}
    for key, population in design['strata'].items():
        sample = [r for r in results if r['stratum'] == key and r['requested_gap'] == gap]
        assert len(sample) == 3
        ratios[key] = sum(r['wall_seconds'] for r in sample) / sum(r['seconds'] for r in sample)
    seconds = sum(s['historical_seconds'] * ratios[k] for k, s in design['strata'].items())
    mature = [w for w in windows if w['date'] >= '2025-02-28']
    assert len({w['date'] for w in mature}) == 28
    mature_per_day = sum(w['seconds'] * ratios[stratum(w)] for w in mature) / 28
    early = sum(w['seconds'] * ratios[stratum(w)] for w in windows if w['date'] < '2025-02-17')
    stages = [s for r in results if r['requested_gap'] == gap for s in r['metadata']['stages']]
    assert all(s['status'] == 0 and s['max_constraint_residual'] <= 1e-6
               and s['max_overlap_kwh'] <= 1e-6 and s['max_binary_residual'] <= 1e-6 for s in stages)
    rows.append({'requested_gap': gap, 'weighted_55_day_seconds': seconds,
                 'flat_334_day_hours': seconds * 334 / 55 / 3600,
                 'mature_seconds_per_day': mature_per_day,
                 'warmup_plus_mature_334_day_hours': (early + 318 * mature_per_day) / 3600,
                 'planning_range_hours': interval,
                 'max_observed_relative_gap': max(s['mip_gap'] for s in stages),
                 'max_constraint_residual': max(s['max_constraint_residual'] for s in stages),
                 'max_overlap_kwh': max(s['max_overlap_kwh'] for s in stages),
                 'stratum_current_to_historical_time_ratios': ratios})
for row in rows:
    row['estimated_speedup_fraction_vs_paired_zero_gap'] = 1 - row['weighted_55_day_seconds'] / rows[0]['weighted_55_day_seconds']
record = {'scope': '334-day main scenario policy from day one, one job on current machine; excludes controls and report/export',
          'status': 'awaiting_user_decision_no_full_rerun_started', 'rows': rows,
          'method': 'Calibrate each of six historical timing strata by sum(current paired wall time)/sum(historical solver time), then apply to all 7,975 saved windows. Mature extrapolation uses first 16 historical days plus 318 times the adjusted final-28-day mean.',
          'range_type': 'Engineering planning allowance, not a statistical confidence interval',
          'uncertainty': ['18 duration-quantile windows, not a randomized annual sample',
                          'Only February/March states and scenarios observed; later seasons and changed incumbents may alter search complexity',
                          'Historical runs overlapped other jobs; paired gap-zero timing separates current speed from gap relaxation',
                          'Gap relaxation can change the subsequent state and future MILP instances',
                          'Any later unaccepted solve can still stop the run at the unchanged 60-second stage cap'],
          'report_and_excel_planning_minutes': [15, 30],
          'report_and_excel_time_is': 'Untimed workflow allowance, conditional on complete validated 334-day replay',
          'stress_results': [{'gap': r['requested_gap'], 'success': r['success'], 'seconds': r['wall_seconds'],
                              'last_status': r['stages'][-1]['status'], 'last_gap': r['stages'][-1]['mip_gap']} for r in stress]}
(OUT / 'estimate.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
table = '\n'.join(f'| {r["requested_gap"] * 100:g}% | {r["flat_334_day_hours"]:.2f} 小时 | '
                  f'{r["warmup_plus_mature_334_day_hours"]:.2f} 小时 | '
                  f'{r["planning_range_hours"][0]:g}—{r["planning_range_hours"][1]:g} 小时 | '
                  f'{r["estimated_speedup_fraction_vs_paired_zero_gap"] * 100:.1f}% |' for r in rows)
text = f'''# 严格互斥重跑：运行时间评估，等待用户决定

原主方案和点预测对照已暂停。主方案55个完整日（2025-02-01至03-27）、点预测对照284个完整日（至11-11）和另一对照22个完整日均已归档；已保存日末SOC、上一段净功率、逐段原始数组、午夜计划、场景及求解日志。正在进行的半日状态仅保留在暂停进程内存中，持久化检查点以完整日为准。

## 决策所需时间

下表估计在本机单独运行主场景策略，从2月1日开始重新计算全部334日所需时间。所有档位都保留充放电二进制严格互斥及原物理容差，完整新实验尚未启动。

| 请求的相对MIP gap | 简单55日均值外推 | 计入后期16路径规模后的外推 | 建议预留的主方案时间 | 比同机零差距计时快 |
|---|---:|---:|---:|---:|
{table}

建议预留范围是工程估计，不是统计置信区间。它以场景规模稳定后的外推为依据，向上预留季节、状态变化和文件处理时间。全年回放通过核验后，Excel和报告整理另预留15—30分钟；这项是工作时间预算，未作为求解性能实测。表中不包含另外两项对照的补跑或失败处理。若优先缩短耗时，可选0.1%；0.05%在本次加权估计中仅比0.1%多约{(rows[1]['warmup_plus_mature_334_day_hours'] - rows[2]['warmup_plus_mature_334_day_hours']) * 60:.0f}分钟，未来实际差异仍可能变化。

## 实测依据

从主方案已完成55日的7,975个求解窗口中分出六层：午夜，以及原滚动耗时小于0.2秒、0.2—0.75秒、0.75—2秒、2—7秒、至少7秒。每层按耗时1/6、1/2、5/6分位取一个已保存输入，共18个窗口。各窗口轮换执行0、0.05%、0.1%三档，共54次两层求解；重新使用完全相同的状态、场景、价格和已锁定普通计划。所有样本通过两层物理与互斥检查。原实验源码运行前后SHA-256一致。

逐层采用“本次样本总耗时 / 相应历史样本总耗时”校准该层全部历史耗时，再累加外推。后期规模外推保留前16日估计，以最近28日的校准日均值估计后318日。这样避免把供体较少的早期速度直接用于全年。原运行曾与对照并行；重新计时的零差距组用于避免将资源释放带来的提速误记为gap收益。

## 仍然存在的具体停止条件

“点计划＋场景执行”对照在2025-02-23 01:20的原失败输入另作压力检验，不参与主方案耗时外推。0.1%和0.05%均在第二层60秒上限停止，结束相对gap约85.126%，候选未执行。首次候选的物理残差约7.66e-5，超过原1e-6标准；相同模型以更紧整数容差复算后仍未达到要求。因此放宽这两档不能解决该已知对照窗口，不能承诺完整对照必然跑完。主方案抽样没有出现此失败，但未来窗口仍可能触发原停止规则。

这里的MIP gap是每次两层目标的求解最优性差距。原第二层费用让步δ=0.001和δ_exec=0.001保持不变；它们与求解gap含义不同。任一数值都不直接保证全年实际电费相对某个全年最优策略的误差上限。

## 文件与复核

- `design.json`：抽样方案、每层数量、预测哈希。
- `results.json`、`timings.csv`：54次主方案样本计时及两层核验。
- `stress.json`：两个额外压力窗口的真实停止记录。
- `estimate.json`：估计公式、原值及不确定性。
- `original_code_hashes_before.json`与`original_code_hashes_after.json`：原源码未变。
- `../paused-before-gap-review/saved-progress.zip`：此前进度归档。

新增适配器只覆盖求解器的mip_rel_gap选项，原实验文件、原零差距适配器、物理约束、两层目标、场景参数均未修改。计时脚本没有推进新的全年策略，也没有把这些样本混入正式购电账单。
'''
(OUT / 'README.md').write_text(text)
print(json.dumps({'rows': rows, 'stress': record['stress_results']}, ensure_ascii=False))
