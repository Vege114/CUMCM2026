"""Summarize only a complete, independently audited annual development run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import battery_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008/budget_feedback/physical334_blend_cap8_grid200'
HOLD1 = ROOT / 'data/results/exp008/mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days/dispatch.npz'
EXP006 = ROOT / 'data/results/exp006/primary/dispatch_2.npz'


def load(path):
    with np.load(path) as z:
        return {key: z[key].copy() for key in z.files}


def run():
    summary = json.loads((OUT / 'summary.json').read_text())
    audit = json.loads((OUT / 'independent_audit.json').read_text())
    assert summary['completed_days'] == audit['completed_days'] == 334 and audit['passed']
    new = load(OUT / 'dispatch_2.npz')
    arrays = {'exp006': load(EXP006), 'hold1_fixed_mask': load(HOLD1), 'budget_DP': new}
    rows = []
    for label, data in arrays.items():
        np.testing.assert_array_equal(data['actual'], new['actual'])
        np.testing.assert_array_equal(data['price'], new['price'])
        assert data['states'][0, 0] == new['states'][0, 0]
        metrics = battery_metrics(data)
        rows.append(dict(label=label, total_cost=float(data['fees'].sum()),
            planned_cost=float(data['fees'][:, :, 0].sum()), emergency_cost=float(data['fees'][:, :, 3].sum()),
            episodes=metrics['charge_starts'] + metrics['discharge_starts'], **metrics))
    frame = pd.DataFrame(rows).set_index('label')
    frame.to_csv(OUT / 'matched_full_comparison.csv')
    dates = pd.date_range('2025-02-01', '2025-12-31')
    monthly = []
    for month in range(2, 13):
        ids = np.flatnonzero(dates.month == month)
        for label, data in arrays.items():
            monthly.append(dict(month=month, label=label, days=len(ids),
                total_cost=float(data['fees'][ids].sum()), planned_cost=float(data['fees'][ids, :, 0].sum()),
                emergency_cost=float(data['fees'][ids, :, 3].sum()),
                throughput_kwh=float((data['charge'][ids] + data['discharge'][ids]).sum()),
                initial_soc=float(data['states'][ids[0], 0]), final_soc=float(data['states'][ids[-1], -1])))
    pd.DataFrame(monthly).to_csv(OUT / 'monthly_comparison.csv', index=False)
    base, prior, candidate = [frame.loc[key] for key in ('exp006', 'hold1_fixed_mask', 'budget_DP')]
    delta_cost = float(candidate.total_cost - prior.total_cost)
    reduction = float(100 * (1 - candidate.total_cost / base.total_cost))
    target = .92 * float(base.total_cost)
    differences = dict(reduction_vs_exp006_pct=reduction, target_8pct_yuan=target,
        gap_to_8pct_yuan=float(candidate.total_cost - target), delta_cost_vs_hold1_yuan=delta_cost,
        cost8pct_and_reversals_gate_passed=bool(candidate.total_cost <= target + 1e-6
            and candidate.direction_reversals < base.direction_reversals),
        episodes_reduced_vs_exp006=bool(candidate.episodes < base.episodes),
        active_slots_reduced_vs_exp006=bool(candidate.active_slots < base.active_slots),
        throughput_reduced_vs_exp006=bool(candidate.throughput_kwh < base.throughput_kwh),
        final_soc_difference_vs_exp006=float(candidate.final_soc - base.final_soc),
        final_soc_difference_vs_hold1=float(candidate.final_soc - prior.final_soc),
        actual_bill_never_includes_internal_terminal_value=True)
    records = json.loads((OUT/'planning_audit.json').read_text())
    gaps = np.asarray([r['mip']['mip_gap'] for r in records])
    differences.update(MIP_gap_mean=float(gaps.mean()),MIP_gap_max=float(gaps.max()),
        MIP_gap_over005=int(np.sum(gaps>.005)),MIP_gap_over01=int(np.sum(gaps>.01)),
        greedy_refinement_success_days=sum(bool(r['greedy_refinement']['success']) for r in records),
        greedy_refinement_iterations_limit=120)
    hgb_path=ROOT/'data/results/exp008/budget_feedback/physical334_hgb_cap8_grid200'
    if (hgb_path/'independent_audit.json').exists():
        hgb_audit=json.loads((hgb_path/'independent_audit.json').read_text())
        if hgb_audit['passed'] and hgb_audit['completed_days']==334:
            hgb=load(hgb_path/'dispatch_2.npz');hm=battery_metrics(hgb)
            np.testing.assert_array_equal(hgb['actual'],new['actual'])
            np.testing.assert_array_equal(hgb['price'],new['price'])
            assert hgb['states'][0,0]==new['states'][0,0]
            differences['same_DP_original_HGB_comparison']={
                'total_cost_delta':float(new['fees'].sum()-hgb['fees'].sum()),
                'direction_reversal_delta':float(candidate.direction_reversals-hm['direction_reversals']),
                'active_slots_delta':float(candidate.active_slots-hm['active_slots']),
                'throughput_delta':float(candidate.throughput_kwh-hm['throughput_kwh']),
                'endSOC_delta':float(candidate.final_soc-hm['final_soc'])}
    (OUT / 'matched_findings.json').write_text(json.dumps(differences, ensure_ascii=False, indent=2))
    keys = [('总费用，元', 'total_cost'), ('计划购电费用，元', 'planned_cost'),
        ('紧急购电费用，元', 'emergency_cost'), ('非空方向反转', 'direction_reversals'),
        ('连续充/放活动段数', 'episodes'), ('活动十分钟时段数', 'active_slots'),
        ('充放电总吞吐，kWh', 'throughput_kwh'), ('等效满循环', 'equivalent_full_cycles'),
        ('功率总变差，kW', 'power_ramp_total_kw'), ('期末 SOC，kWh', 'final_soc')]
    table = ['| 指标 | exp006 | 同混合预测 固定掩码 hold1 | 实时预算 DP |', '|---|---:|---:|---:|']
    for label, key in keys:
        table.append(f'| {label} | {base[key]:,.4f} | {prior[key]:,.4f} | {candidate[key]:,.4f} |')
    status = ('费用降幅和非空方向反转达到这两个数值门槛；全任务的其他题目及最终交付仍由主任务核对。'
        if differences['cost8pct_and_reversals_gate_passed'] else '尚未达到费用至少下降 8% 且减少非空方向反转的联合门槛。')
    readme = f'''# 同混合预测 预测的年度实时换向预算 DP

2025-02-01 至 2025-12-31 共 334 日、48,096 个十分钟时段连续运行完毕，并通过独立审计。实际费用较 exp006 下降 **{reduction:.6f}%**；较同混合预测 的固定掩码 hold1 方案费用差额为 **{delta_cost:,.4f} 元**。{status}

{chr(10).join(table)}

实际计费只包括题目规定的计划购电及 5 倍紧急购电费用。0.002 元/kWh 吞吐项和终端库存估值仅用于内部规划，未加入或减去上表费用。距 exp006 费用下降 8% 的门槛差额为 {differences['gap_to_8pct_yuan']:,.4f} 元（正值表示仍高于门槛）。活动段数、方向反转、活动时长和吞吐是不同指标，不能用任一项替代其他项或据此声称电池寿命增长。

每日日初沿用本候选上一日的真实 SOC 和最后非空真实方向，重新求解 3 条历史路径的物理 MIP，再以全 28 条历史路径进行原固定掩码贪心目标的 120 次上限购电优化。日前 Q 锁定后，预算 DP 根据当前观测决定充放动作；其状态包括 SOC、误差档、上一非空方向及当日剩余 0–8 次换向。200 kWh 网格、3 误差档、每档 3 个尾部代表点和所有数值参数在运行前锁定。五日诊断固定 Q，没有 Q 块修正；本次年度也不加入 Q 块修正。

该 Q 初始化算法依然假定一个固定模式掩码，而实际 DP 具有状态依赖的换向时机。此处保留并明确披露这个近似，没有宣称日前 Q 与最终反馈策略已经精确联合优化。3 情景 MIP 的路径补救也是乐观代理；MIP gap 平均 {100*differences['MIP_gap_mean']:.6f}%、最大 {100*differences['MIP_gap_max']:.6f}%；购电细化成功日数 {differences['greedy_refinement_success_days']}/334，上限仍为120次；完整信息保存在逐日审计中。相同5秒时限下求解器可能返回不同可行解，不宣称逐值确定重演。

本次实际方向预算直接在每个时段执行：空闲不改变上一方向，午夜第一次反向也会扣减当日预算。每天预算从 8 开始且始终非负，所以含午夜反向在内的年度上界是 2,672 次。计划掩码的方向状态没有冒充真实方向；上一真实方向和真实 SOC 都跨日传递。

独立审计核对 {audit['source_closure_files']} 个本地导入源码的启动归档和完整运行时闭包，核对所有原 HGB、ExtraTrees 两族模型及固定原始等权混合来源文件、全部 334 日已发布预测和 28 条完整历史误差，重建每日 DP 模型及值表并逐元素复现实际流量。所有 334 个 MIP 的物理和模式约束均核对；{audit['future_actual_prefix_mutations']} 次非对称未来净负荷扰动保持过去动作、SOC、方向和剩余预算不变；固定 5 日的 Q 优化结果精确复现，并通过当前/未来实际及未发布未来预测扰动。最大物理残差为 {audit['max_physical_error_kwh']:.3g} kWh。第一批 3 日只按可行性和因果性过门，随后继续相同运行，已确认首批结果和计划文件完全保留。

`matched_full_comparison.csv`、`monthly_comparison.csv` 和 `matched_findings.json` 保留费用、库存及强度比较。新方案与 exp006 的末端库存差额是 {differences['final_soc_difference_vs_exp006']:,.4f} kWh；与 hold1 的末端库存差额是 {differences['final_soc_difference_vs_hold1']:,.4f} kWh。全部结果属于反复使用同一开发年度后的开发评估，不是未触碰的独立测试，也不是最终总报告。
'''
    (OUT / 'README.md').write_text(readme)
    print(json.dumps(differences, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
