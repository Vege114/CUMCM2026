## 8. 复现说明

原实验源码不变，新增入口位于`reports/experiments/exp005/soft_penalty/`。两组独立命令分别为：

```sh
.venv/bin/python reports/experiments/exp005/soft_penalty/run.py --beta 0.1 --out .work/reproduce-beta-0.1
.venv/bin/python reports/experiments/exp005/soft_penalty/run.py --beta 0.01 --out .work/reproduce-beta-0.01
```

输出目录必须不存在。每个完整日保存场景、午夜方案、原始十分钟执行、两层求解日志和所有超过阈值的重叠节点；progress.json保存下一日初始状态。逐日诊断不净化充放电数组。

重建报告：`.venv/bin/python reports/experiments/exp005/soft_penalty/build_report.py --complete --build`。完整证据表位于soft_penalty/evidence，费用与图表使用同一份原始数据。原连续LP与严格互斥报告分别归档于versions/lp-failure和versions/strict-paused。

原模型11项测试已通过，覆盖单段五倍电价手算、共享节点后分支、满电硬爬坡的重叠反例、D5-A紧急充电、未来实际值扰动不改变之前动作、历史供体可见性与锁定普通计划。全年独立回放审计补充跨日跨月状态、实际结算与各层最优状态核对。

Excel复现：先运行`soft_penalty/prepare_workbook.py`生成主方案表格载荷，再用未修改的`reports/export_workbooks_v2.mjs`通过artifact-tool导出，最后执行`soft_penalty/verify_workbook.py`回读核验。严格互斥旧版完整日另存于`data/results/exp005/paused-before-gap-review/saved-progress.zip`。

audit（原值可查对应CSV）

| beta | check | value | limit | passed | check_label |
| --- | --- | --- | --- | --- | --- |
| 0.1 | balance | 2.16005e-12 | 1e-06 | True | 逐段供需平衡残差 / kWh |
| 0.1 | soc | 1.81899e-12 | 1e-06 | True | SOC递推残差 / kWh |
| 0.1 | bounds | 9.09495e-12 | 1e-06 | True | 电量与SOC边界超出 / kWh |
| 0.1 | fixed_grid | 0 | 1e-06 | True | 日内普通计划变化 / kWh |
| 0.1 | actual_source | 0 | 1e-06 | True | 实际输入与冻结原值差 / kW |
| 0.1 | cross_day_soc | 0 | 1e-06 | True | 跨日SOC断点 / kWh |
| 0.1 | ramp | 1,000 | 1,000.000001 | True | 相邻净功率最大变化 / kW |
| 0.1 | node_constraints | 1.20622e-10 | 1e-06 | True | 规划节点最大约束残差 |
| 0.1 | cost_budget | 4.36557e-11 | 1e-06 | True | 第二层超预算金额 / 元 |
| 0.1 | fees | 4.36557e-11 | 1e-06 | True | 独立账单差额 / 元 |
| 0.1 | integer_variables | 0 | 0 | True | 整数变量数 |
| 0.01 | balance | 2.95586e-12 | 1e-06 | True | 逐段供需平衡残差 / kWh |
| 0.01 | soc | 1.04308e-11 | 1e-06 | True | SOC递推残差 / kWh |
| 0.01 | bounds | 2.00089e-11 | 1e-06 | True | 电量与SOC边界超出 / kWh |
| 0.01 | fixed_grid | 0 | 1e-06 | True | 日内普通计划变化 / kWh |
| 0.01 | actual_source | 0 | 1e-06 | True | 实际输入与冻结原值差 / kW |
| 0.01 | cross_day_soc | 0 | 1e-06 | True | 跨日SOC断点 / kWh |
| 0.01 | ramp | 1,000 | 1,000.000001 | True | 相邻净功率最大变化 / kW |
| 0.01 | node_constraints | 3.75557e-10 | 1e-06 | True | 规划节点最大约束残差 |
| 0.01 | cost_budget | 3.63798e-11 | 1e-06 | True | 第二层超预算金额 / 元 |
| 0.01 | fees | 4.36557e-11 | 1e-06 | True | 独立账单差额 / 元 |
| 0.01 | integer_variables | 0 | 0 | True | 整数变量数 |
