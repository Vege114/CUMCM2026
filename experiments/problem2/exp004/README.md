# exp004：只改预测的短序列与季节性实验

从 `codex/q2-optimization` 的 `c2fb69d` 新建 `codex/q2-discussion-seasonality`。讨论稿来自 `reports/templates/q2-discussion/q2-discussion-draft.json`。用户后续明确只做预测，规划组负责决策。本次没有修改 exp003 规划、预算和储能执行，也没有实现讨论中的随机规划、滚动储能或跨日预看。

## 结果与使用建议

主种子42，2025年2–12月334日，同一原调度器：

| 预测组 | 总购电费/元 | 紧急购电量/kWh | 口径 |
|---|---:|---:|---|
| no_season | 13,611,587.8329 | 160,912.1654 | 可实施对照 |
| causal_season | 13,626,145.4060 | 162,942.3891 | 事先指定正式方案 |
| oracle_season | 13,572,926.4786 | 145,611.6324 | 全年季节信息，含未来，不能正式排名 |

无季节组比exp003正式少1,377,034.8153元（9.1872%）。加历史季节后反而多14,557.5731元，另外两种子方向一致。常规MAE/RMSE/WAPE并未优于exp003：费用代理损失倾向多预测负载、少预测PV，带来更充分的日前购电。只能把费用改善归于整个预测组合，不能宣称精度整体改善或单个模块有效。

规划组可以优先用 `no_season`，保留 `causal_season` 作为本轮冻结正式组和季节消融；这不改变历史记录中的正式角色。

## 数组接口

```python
from experiments.problem2.exp004.predict import ForecastStore

store = ForecastStore('no_season', seed=42)
forecast = store.get(31 * 144)  # 2025-02-01 00:00
# float64 shape (144, 2)，列顺序 load、pv，单位 kW
net_kw = forecast[:, 0] - forecast[:, 1]
paths = store.completed_error_paths(100 * 144, limit=28)
# origins: 已发布且整天已揭晓的历史午夜；errors_kw: 实际−当时预测
# shape (n,144,2)，保留槽位和负载/PV联合依赖；早期历史不足时n更小，首日为空。
```

origin为从2025-01-01 00:00起的十分钟索引，必须是2–12月的午夜。未来预测不使用当天尚未发生的观测。全年探索必须明确 `ForecastStore('oracle_season', allow_oracle=True)`，此显式选择防止混入正式策略。

数组本身在`data/results/exp004/predictions.npz`，每组各3个种子字段，例如`no_season_seed_42`。`origins`记录334个午夜，`*_base`、`*_residual`、`*_seasonal_shift`和`*_mask`保留主种子解释量。无需训练权重即可读取。查询会校验档案SHA-256并返回副本，修改返回值不会污染缓存。

## 固定方法

168小时序列，两个独立卷积分支，共4930参数；全部完整历史保留，90日半衰期降权；电价加权的5:1非对称Huber损失；每月只用过去7日早停。3组×11月×3种子=99次训练。CPU两线程，最多60轮。季节性是逐槽位年周期一、二阶Fourier加星期项的岭拟合，再计算目标日相对周期参考日的季节变化；历史组严格前缀拟合，全年组拟合365天并标为探索。

详见 [完整方法](../../../reports/experiments/exp004/methods.md)、[决策日志](decision-log.md)、[配置](protocol.json)。

## 复现

仓库根目录，既有 `.venv` 环境：

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_q2_*.py' -v
.venv/bin/python -m experiments.problem2.exp004.train
.venv/bin/python -m experiments.problem2.exp004.evaluate --workers 3
.venv/bin/python -m experiments.problem2.exp004.verify
.venv/bin/python -m experiments.problem2.exp004.export --node /absolute/path/to/codex/node
.venv/bin/python -m reports.build_report_exp004 --complete --node /absolute/path/to/codex/node
```

首次完整重训会生成忽略的 `runs/` 检查点。代码、数据、配置及预测哈希绑定缓存；发生变化时拒绝旧缓存。CPU/GPU、TensorFlow与求解器差异可能引入数值差异；已归档原值是报告的核算依据。MILP日志保留真实gap、超时和回退，未知gap不写零。

`verify`直接检查交付数组，不需要训练权重；99组保存/重载结果为训练时留痕。41项Q2测试通过，9组共432,864槽位通过独立供需、SOC、功率、费用与原因果贪心检查。1145个既有算法及结果文件未变。

## 交付

[八节报告](../../../reports/experiments/exp004/report.md) · [离线交互报告](../../../reports/experiments/exp004/report.html) · [四日完整表1–3](../../../reports/experiments/exp004/specified_dates.md)

三份 `result2.xlsx` 分别放在报告目录的 `no_season/`、`causal_season/`、`oracle_season/`，保留题目模板名。原始精度CSV、逐槽位NPZ、验证清单在`data/results/exp004/`；同源比较数据及PNG/SVG在报告包。历史登记exp001–003保持原样。
