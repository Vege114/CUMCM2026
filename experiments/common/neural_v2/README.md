# exp002：固定轻量网络与多阶段风险调度

正式种子为 42；2026、3407 只用于全年稳定性评价。每月重新训练一个四分支网络，共 33 个正式检查点。第一问继续沿用既有成果。

```sh
.venv/bin/python -m experiments.common.neural_v2.run prepare
.venv/bin/python -m experiments.common.neural_v2.run train
.venv/bin/python -m experiments.common.neural_v2.run predict
.venv/bin/python -m experiments.common.neural_v2.run calibrate --workers 4
.venv/bin/python -m experiments.common.neural_v2.run replay --workers 4
.venv/bin/python -m experiments.common.neural_v2.run export --node /absolute/path/to/codex/node
.venv/bin/python -m experiments.common.neural_v2.run report
```

训练使用现有 TensorFlow/Metal 环境并强制检查 GPU 输出、权重更新和模型保存读取。训练阶段缓存每个检查点的完整月度预测；`predict` 将它们汇总成按发布时刻、目标区间、变量、种子排列的正式档案。风险校准固定使用 1 月 25—31 日；正式费用评价为 2 月 1 日至 12 月 31 日，共 334 天。四个问题分别从共同基线的 1 月末储电量继续运行。

`runs/exp002/` 是可再生本地缓存，不进入 Git。训练缓存签名绑定训练代码、协议与原始数据；回放签名额外绑定所有调度代码、训练签名及旧预测档案。签名不符即拒绝复用，请使用新的 `--run-id`。回放逐日落盘，恢复时核对跨日状态。已保存的原始检查点目录不可删去后再宣称是无需训练的完整恢复；分享包包含预测档案，正式重现全部训练仍需原始数据和 GPU。

调度记录在每日缓存的 `solvers` 中，包含信息截止时间、场景历史上界、求解状态、剩余差距与回退来源。每次场景 MIP 的预算是 2 秒，包含至多 0.5 秒的固定控制器区域 LP；场景组装、确定性候选和物理核验另计时间。获得通过核验的可行解并不代表达到 1% 最优性目标，未知差距保留为 `null`。

`data/results/exp002/` 保存机器可读结果与四个竞赛工作簿；`reports/experiments/exp002/` 保存八部分正文、离线交互报告、PNG/SVG 及独立浏览所需数据。历史 exp001 仅加载已保存预测；重新积分与新物理协议明确记录，不覆盖旧报告或注册记录。

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_neural_v2.py'
.venv/bin/python -m unittest discover -s reports/tests
```

导出工作簿使用 Codex 捆绑的 `@oai/artifact-tool`。`export --prepare-only` 只准备核验后的 JSON/CSV，`export --verify-only` 独立读取成品并重算物理及费用。报告构建不触发任何训练或调度。
