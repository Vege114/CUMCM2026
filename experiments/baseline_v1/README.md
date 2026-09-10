# 可运行基础模型 v1

先看 [运行报告](REPORT.md)。本目录集中管理算法、配置、运行产物和图表，模型运行只依赖 Python 3.10+ 标准库；生成图表另需 `requirements-report.txt`。

| 文件 / 目录 | 职责 |
| --- | --- |
| `run.py` | CLI、后端选择、现实时间保护、运行元数据、错误归档 |
| `baseline/config.py` | 参数定义与覆盖条件校验 |
| `baseline/geometry.py` | 不依赖模拟器的凸几何、直径、包围圆、光学覆盖 |
| `baseline/planning.py` | 全局巡检点、第二测点候选区域及代价函数 |
| `baseline/strategy.py` | 搜索—定位—清除状态机、目标调度、失信号回退 |
| `baseline/protocol.py` | HTTP、幂等重试、脱敏事件日志 |
| `baseline/mock.py` | 独立自建离线环境，非官方随机生成器 |
| `configs/default.json` | 可选策略参数覆盖；命令行传 `--config` |
| `tests/` | 几何、覆盖、协议、边界场景测试 |
| `benchmark_suite.py` | 固定种子离线套件；已有运行目录不会覆盖 |
| `evaluate_runs.py` | 调用全局 Benchmark，生成每局指标和比较索引 |
| `runs/` | 每局完整脱敏日志、配置、指标、界面证据；已核验基线纳入 Git |
| `make_report.py`、`report_template.md` | 可复现报告及科研图表生成 |
| `REPORT.md`、`figures/`、`report_data.json` | 本次报告、PNG/SVG 图表和图表源数据 |

```powershell
# 在仓库根目录执行
python experiments/baseline_v1/run.py --backend mock --problem 4 --name my_first_q4
python -m unittest discover -s experiments/baseline_v1/tests -v
python experiments/baseline_v1/benchmark_suite.py
python experiments/baseline_v1/evaluate_runs.py
```

真实模拟器需先由界面启动相应问题的演练并等待就绪，再设置本进程环境变量 `JAMMERS_ROBOT_ID` 后运行 `--backend official`。CLI 中的 `official` 表示连接官方软件，元数据仍标记为 **practice 演练**；本版没有自动启动正式测试的能力，也没有正式次数管理。不要用该入口把正式测试误记成演练。

调参优先修改 JSON 中的 `movement_weight`、`max_local_measurements`、`directional_grid_m`、`optical_grid_m`；候选点排列在 `measurement_candidates()`，目标排序在 `Strategy.run()`。1800 m 目标圆、1000–1500 m 接收范围、20 m 清除半径等属于题给物理条件，连接官方软件时应保持原值。运行目录和源码哈希一起保存；比较版本时使用新的目录前缀，避免把旧证据误当成新结果。

题目指标与校验器见 [Benchmark](../../Benchmark/README.md)。未核验的新实验和本机私有配置可放在仓库忽略的 `experiments/runs/` 或 `config/*.local.*`；不要提交队号、密码、模拟器数据库和登录状态。
