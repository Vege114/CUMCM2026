# exp005：严格互斥重跑

当前报告入口为report.html与report.md，仍按八节实验报告模板编写。原连续LP失败报告保存在versions/lp-failure；原实验源码未修改。

新增实现位于strict_rerun：strict_backend.py在求解入口增加每节点0/1互斥，run.py调用原回放逻辑，build_report.py独立核验并重建报告。午夜和执行阶段仍只有费用、平稳性两个目标。物理残差超过1e-6的候选解不执行；同层可以在剩余60秒内收紧数值精度复算相同模型，不增加第三个目标。

已完成55个完整日（2025-02-01至2025-03-27），共7,920个十分钟区间。按用户要求暂停计算，已完成日期及下一日初始状态已单独保存；0.1%和0.05%最优性差距的抽样计时结果见第八节。正式新实验尚未启动，等待用户决定。

当前运行记录：data/results/exp005/strict-mutual-exclusion-certified。strict-mutual-exclusion为默认整数容差预检；strict-mutual-exclusion-tight为全局高精度预检。各次真实状态与费用独立保存，不混算。

仅在完整334日通过核验后生成正式result2.xlsx。严格重跑、测试及报告重建命令见报告第八节。
