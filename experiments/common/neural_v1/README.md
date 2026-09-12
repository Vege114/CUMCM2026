# 首轮神经网络预测与基础调度

在仓库根目录使用锁定环境：

```bash
uv sync --locked
uv run --locked python scripts/check_environment.py
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m experiments.common.neural_v1.train
uv run --locked python -m experiments.common.neural_v1.complete_validation
uv run --locked python -m experiments.common.neural_v1.evaluate
uv run --locked python -m experiments.common.neural_v1.export
uv run --locked python reports/build_report.py --experiment exp001
```

正式训练默认要求 GPU。NVIDIA Windows 主机使用 WSL2 环境，安装和运行说明见 [机器学习与 CUDA 环境](../../../docs/ml-environment.md)。PowerShell 中用 `scripts/run_ml.ps1 tf` 替代上述命令的 `uv run --locked` 前缀。CUDA 环境适配已更新训练源代码签名，开始新训练请指定新的 `--run-id`，避免接续旧签名的检查点。Apple Silicon 继续使用 Metal。

`protocol.json` 固定数据时标、计费、训练切分、随机种子和选择规则。训练从头拟合每个月的参数，不把当月实际值用于训练。四个预测目标采用相互独立的网络分支：负载、历史光伏、电价和预报修正；各分支不读取其他变量的数值特征。所有分支仅共享已知的日历和提前量。

训练的 `runs/exp001/` 保存 165 组按月份、模型与随机种子编号的 `.keras` 权重、`.npz` 预测和 `.json` 训练元数据。再次执行会校验代码与输入签名后恢复，代码或数据改变时应使用新的 `--run-id`，避免混合实验。权重与缓存不提交 Git；正式结果和训练元数据由评估程序另存至 `data/results/exp001/`。

快速验证完整网络可运行：

```bash
uv run --locked python -m experiments.common.neural_v1.train --run-id smoke-new --months 2 --variants gru --seeds 42 --epochs 2
```

正式评估包括三个种子的各自预测和调度成绩、种子均值的调度、两种历史基线、特征消融、原始预报对照、四种日内预报时刻组合和已知价格对照。月初选模只比较此前七个完整日期的调度总费用，验证起点使用共同历史基线的储电量；部署时使用实际连续传递的储电量。

本轮采用最基础的确定性计划与固定实时执行规则，没有额外风险缓冲、分布鲁棒优化或日末储能价值项。不同方案的期末储电量同时报告；预测精度与总费用可能不同向变化。
