## 8. 复现说明

严格重跑命令：`.venv/bin/python reports/experiments/exp005/strict_rerun/run.py --out .work/exp005-strict-reproduce`。输出目录须不存在；默认请求334日。原源码运行前后SHA-256保存在运行目录，新增求解适配器位于strict_rerun/strict_backend.py。

新增测试：`.venv/bin/python -m unittest discover -s reports/experiments/exp005/strict_rerun -p test_strict_backend.py -v`。报告重建：`.venv/bin/python reports/experiments/exp005/strict_rerun/build_report.py --complete --build`，仅在回放完成或明确停止后使用complete。

原LP报告归档于versions/lp-failure，原实验源码和失败输入保留。新证据包括每日午夜方案、逐段真实轨迹、两层求解日志、同输入互斥对照、独立物理/结算审计、代码哈希及测试日志。报告完成状态与策略是否完成334日评价分别标记。

两项对照在上述命令中分别加入`--mode point`、`--mode point_scenario_control`，并使用独立的新输出目录。

**暂停后的运行时间评估。** 原主方案55个完整日和点预测对照284个完整日已归档，完整日状态可以复核；未完成日仅保留在暂停进程内存。正式近似重跑没有启动。从已保存主方案选18个分层窗口，三档共54次两层求解全部通过原互斥和物理标准。按本机单独运行、从头重算334日主方案估计：0.1%预留4—6小时，0.05%预留4.5—6.5小时，原零差距约5—7小时；Excel和报告另预留15—30分钟。这是有限样本的工程估计，不是保证完成的时限，也不包含另外对照的处理。

最优性差距仅改变每次求解的停止精度，二进制互斥、原物理容差和两层费用让步保持不变，不是全年实际电费误差上限。原点计划＋场景执行对照的失败窗口在0.1%和0.05%下仍于60秒停止，相对gap约85.126%，未执行失败候选。详细抽样、加权外推和压力检验见`data/results/exp005/gap-timing-review/README.md`；保存进度见`data/results/exp005/paused-before-gap-review/saved-progress.zip`。
