# 最终报告预测证据

本目录只从已冻结的逐日预测重新统计指标，不拟合模型、不选择参数、不运行调度。

`annual_metrics.csv`、`monthly_metrics.csv`、`lead_metrics.csv` 是报告主数据。两种模型分别为 Q2 最终融合与后续问单 HGB；均在午夜发布的未来144槽、2025-02-01至12-31评分。它不代表第三、四问所有其他发布时间的评分。每份文件都有负载、光伏、净负荷，及全天 / 实际PV>0两种总体。`n=0`或实际绝对值分母为0时指标留空，不填零。

`history_midnight_comparable.csv` 区分已有重评分、午夜正式评分、exp005原登记四舍五入值与本次冻结预测的只读评分。exp001/002原登记的多次发布总体不能与午夜评分混用；全部原记录另存 `history_original_registry.csv`。exp005仅保留原登记中真实存在的4项，发电时段load/net为空；来源为固定git提交而非当前main。exp007已排除。

`model_configuration.json` 给出两类树模型、31个特征、超参数、seed42及先raw融合后Ridge28/非递归记忆的顺序。`runtime_evidence.csv` 保留22个原始主模型fit加紧随的验证集predict计时之和及原pre-bridge整体计时；该timer并非纯训练时间，纯fit、正式预测及后处理分阶段耗时均缺失，不把缺失当0。仅有一个种子，标准差不估算。`training_models_and_boundaries.csv`含44模型的原文件hash和逐月训练/验证边界。

所有2025正式期已被用于开发比较，不能称为未触碰的独立测试集。最终Q2费用实际下降6.1443%，原8%门槛已由用户接受当前结果的指令解除；本证据不宣称达到8%。后续问保留各自已完整核验的单HGB模型，不能把Q2融合收益直接外推至Q3/Q4。

复现：`.venv/bin/python -m experiments.exp008.final_forecast_evidence`。脚本拒绝覆盖已有目录；先审查已有证据再决定是否另存重生成。图为PNG/SVG，直接读取本目录CSV。
