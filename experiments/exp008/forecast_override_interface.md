# 预测覆盖接口与三日对照（未运行全年）

`run.run_case(..., forecast_override=None)` 保留原默认 `UnifiedForecasts(seed, calibration)`。覆盖时接收同样提供 `data`、`seed`、`store.values/origins`、`get(day,slot,scenario)`、`_observed`、`net_error_paths` 的对象，并要求 `forecast_identity()` 返回明确的 `model_id` 与预测值、预测起点的 SHA256。

`AuditedForecastOverride` 可包裹现有 `JointForecasts`，无需改动网络、校准或物理模块。`run.py` 独立重算实际 store 的 hash，拒绝声明与加载值不一致、seed/calibration 冲突，以及执行过程中混入其他模型；覆盖身份进入 `config.forecast_override` 和 cache signature。有效校准另记 `config.effective_forecast_calibration`。同名 case 若模型/代码改变会拒绝覆盖，应使用新的 case 名。

```python
from experiments.exp008.forecast_override_adapter import AuditedForecastOverride
from experiments.exp008.neural_joint_dispatch import JointForecasts
from experiments.exp008.planner import Settings
from experiments.exp008.run import run_case

forecast = AuditedForecastOverride(
    JointForecasts(),
    model_id="exp008_joint_shared_cnn_seed42_then_fixed_ridge28",
    source_paths=[
        "data/results/exp008/neural_joint_calibration/joint_ridge28.npz",
        "data/results/exp008/neural_joint_calibration/manifest.json",
        "data/results/exp008/neural_joint/manifest.json",
    ],
)
run_case("explicit_new_case_name", "3", days=3, issued_residuals=True,
         settings=Settings(future_shortfall_weight=1.5), deadband=20.,
         forecast_override=forecast)
```

每个发布时刻归档当前 load/PV/price 的数组 hash、同发布时间历史净误差与价格误差的 hash、实际模型 ID 和原预测 store hash。`issued_error_paths` 接收同一个覆盖对象并重新计算历史误差，不读取旧模型 residual cache。外层 residual audit 的 `source` 明确指向所选覆盖模型；一月周期冷启动单独保留，不能标成联合 CNN。

Q3/Q4-3 的 PV 仍是附件 3 当前发布版本的因果校准，价格机制也不变。因此这次覆盖改变的是共同午夜负载轨迹、日内负载修正及其历史净误差；不能声称三种预测通道全部改变。此对照将原无校准网络管线换为 Joint + Ridge28 管线，不能解释成仅网络结构的净效应。

## 已完成的有限验证

证据在 `data/results/exp008/forecast_override_pilot/v2/summary.json`。修改前 `run.py` 原文与 hash 保存在其父目录；最初测试辅助代码对只读 property 赋值失败的记录保留在父目录，已修正为隔离数据副本，v2 为通过记录。

5 项测试全部通过：

1. Q3/Q4-3 各 3 日，修改前默认与新接口无 override 的 11 个数组严格相同（零差异）。
2. 新默认与既有 `update_value_diagnostic/future1.5_d334_issued` 年度档案的前三日全部数组严格相同。
3. 24 个发布点的当前预测与历史残差同时切换，历史误差可由同模型历史预测逐项重构，所有标签都来自此前完整日。2 月 1 日历史仍全部是一月周期冷启动，因此当日历史残差不变；2 月 2 日起出现联合模型历史误差。
4. 8 组新建对象的未来扰动检查：修改发布时刻起的所有实际 load/PV/price，以及未发布的官方 PV 版本，当前预测及同发布历史残差严格不变。该检查覆盖接口；神经网络训练与 Ridge28 拟合因果性沿用各自独立 manifest。
5. 修改已加载 store 后身份校验拒绝；同一 case 使用不同模型 ID 时 cache 拒绝；发布记录不再把联合模型残差称为旧冻结 CNN。

6 份三日派发档案均重新对原始附件核验费用、平衡、SOC、跨日连续性、功率和禁止同充放，全部通过。

| 场景 | 原管线费用/元 | Joint + Ridge28 费用/元 | 变化 | 非空方向反转 | 吞吐量/kWh | active slots |
|---|---:|---:|---:|---:|---:|---:|
| Q3 | 131525.50 | 134850.58 | +2.53% | 29 → 43 | 117617.98 → 120866.61 | 351 → 375 |
| Q4-3 | 141220.72 | 143906.66 | +1.90% | 41 → 53 | 123664.61 → 129954.58 | 329 → 360 |

日期固定为 2025-02-01 至 02-03，仅用于接口和有限配对诊断。费用增幅主要来自日前计划削减后出现调整和紧急费用：Q3 新上调费 9077.46 元、紧急费 9349.50 元；Q4-3 分别为 9444.99 元、10292.93 元。新方案末 SOC 也更低（Q3 1437.44 vs 2498.99 kWh；Q4-3 1589.68 vs 2671.83 kWh），没有通过保留更多末库存解释费用上涨。

未据这三日选择新参数，也未扩展全年或选择最终模型。后续负载日能量修正候选可以复用此接口，仍需显式模型身份和独立来源。
