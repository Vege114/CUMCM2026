## 8. 复现说明

代码提交：`46185b82feac52868e16e3e0742808f1c1a278c2`。数据校验值、环境版本、训练参数、选择结果和核验成绩随报告保留。[四个结果工作簿与逐区间档案](https://github.com/Vege114/CUMCM2026/tree/codex/neural-forecasting-v1/data/results/exp001)保存在实验分支；报告网页的数据已完整内嵌，可以独立浏览。下面的复现命令在 codex/neural-forecasting-v1 实验分支执行；上述提交固定训练与评估代码，报告生成器使用随本报告提供的版本。

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

权重保存在被忽略的 runs 目录；代码和输入签名相同才允许恢复。预测归档使用显式发布索引、目标区间索引和目标变量顺序，定义见 prediction_archive.json。正式工作簿为 result2.xlsx、result3.xlsx、result4-2.xlsx、result4-3.xlsx。

所有工作簿的全天购电费表示包含计划、调整、紧急购电的总费；不同工作表的这个值不能再相加。verification.json 记录独立费用复算、能量平衡、连续储电和保存文件读取结果。

技术资料：[Apple Metal 插件](https://developer.apple.com/metal/tensorflow-plugin/)、[TensorFlow 时间序列教程](https://www.tensorflow.org/tutorials/structured_data/time_series)、[门控循环层接口](https://keras.io/api/layers/recurrent_layers/gru/)、[因果卷积层接口](https://keras.io/api/layers/convolution_layers/convolution1d/)、[SciPy 线性规划接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。这些资料说明实现接口，实验结论来自本仓库实测结果。