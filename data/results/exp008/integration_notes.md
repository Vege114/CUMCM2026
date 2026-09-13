# 已验证 Q3 / Q4-3 修正的统一运行器接口

正式入口仍为 `experiments.exp008.run.run_case`，新增可选 `issued_residuals=False`。启用时仅 Q3 / Q4-3 使用 `issued_error_paths(forecasts, day, slot, scenario)`，其余场景保持原午夜路径。`Settings.future_shortfall_weight` 默认 `None`，因此不改变旧目标。

复现已验证组合配置：

```python
run_case(new_case_name, scenario,
         issued_residuals=True,
         settings=Settings(future_shortfall_weight=1.5))
```

CLI 等价配置（正式重跑时使用新的 case 名称）：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp008.run \
  --case integrated_issued_opportunity_v1 --scenarios 3 4-3 \
  --issued-residuals --future-shortfall-weight 1.5
```

`plan` 新增 `next_update_slot=None` 和 `future_shortfall_weight=None` 两个可选参数。前者是**相对当前剩余预测的索引**：36 表示先执行 36 个十分钟槽，再获得下一次信息。统一运行器在每个合法发布点自动传入 `stop-start`。后者优先于 Settings 中的同名参数；没有边界时不激活机会代理。18 点剩余 36 槽，边界恰等于预测长度，因此全段仍按 5p；关闭日内更新或运行 Q2 / Q4-2 时也不启用代理。真实结算一直由原 `settle` 按题目账单计算。

规划 metadata 保留 `nonanticipative_recourse_certificate=False`，增加实际使用的 `planning_emergency_weights`、边界、未来权重和代理是否激活。该 LP 仍允许各历史情景拥有不同未来储能状态，是明确标注的乐观规划代理，不是完整非前视情景树。执行保持原因果反馈与物理检查。

新增参数写入 config；签名结合完整 config 与当前 `experiments/exp008/*.py` 的哈希。修改任一选项后重用已有 case 会拒绝覆盖。旧归档保持不动，试验应使用新 case 名。

本次没有重跑全年。仅运行新配置和旧默认配置在两问上的各 3 日，共四臂。新配置逐数组匹配独立 `future1.5_d334_issued` 归档前 3 日；默认配置逐数组匹配 `joint_initial` 前 3 日。全部 11 组数组最大差为 0，四臂独立物理/源数据/账单/时点核验通过。切换 `issued_residuals` 或未来权重后复用已有 case 均被拒绝，已有调度文件哈希不变。证据：`integration_regression.json`。

12 个相关单元测试通过：新机会权重数值回归 3 项、同刻残差因果回归 3 项、原规划/物理/梯度回归 6 项。没有修改 `mode_planning.py` 或其他代理正在维护的执行机制。
