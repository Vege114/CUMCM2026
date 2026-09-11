# 改进实验 v2

针对 [v1 报告第 7 节](../baseline_v1/REPORT.md) 的四项问题，完成了定向可接收性建模、覆盖与路径优化、联合调度及分层配对验证。**本版在 macOS 完成代码和离线验证，尚未运行官方 Windows 模拟器。** 已归档的结果见 [REPORT.md](REPORT.md) 和 [逐场配对结果](runs/validation_20260911/COMPARISON.md)。

运行只需要 Python 3.10+ 标准库。新代码、参数、测试、运行入口和产物集中在本目录；通过仓库包路径直接引用 v1 的稳定实现，因此迁移到 Windows 时请拉取整个仓库。v1、原题、原始附件、exe 和旧实验数据保持原样。

## 直接运行

以下命令在仓库根目录执行，macOS 将 `python` 换成 `python3`。入口本身支持从任何工作目录用绝对路径调用。

```powershell
# 只检查导入、参数、覆盖计划与源码指纹，不创建运行目录、不调用 HTTP
python experiments/baseline_v2/run.py --problem 4 --dry-run

# 固定种子离线单局；每局使用新名字，禁止覆盖既有证据
python experiments/baseline_v2/run.py --backend mock --problem 4 --seed 20260910 --name my_v2_q4

# 指定分层场景（场景文件决定题型与种子）
python experiments/baseline_v2/run.py --scenario q4_outward_positive --name my_outward_test

# 30 对场景，v1 原默认参数与 v2 同环境重跑；失败保留并返回非零退出码
python experiments/baseline_v2/benchmark_suite.py --prefix my_paired_suite
python experiments/baseline_v2/make_report.py experiments/baseline_v2/runs/my_paired_suite/suite.json
python experiments/baseline_v2/verify_artifacts.py experiments/baseline_v2/runs/my_paired_suite --require-current-source
```

`benchmark_suite.py --resume` 仅复用源码、参数、场景集合一致且已有完整归档的局；不会把失败局剔除，也不会在修改代码后静默跳过旧局。半写入的 `running` 目录应保留排查，换新 prefix 重跑。所有源码和配置的指纹都归档，包括被引用的 v1 模块与独立评价器；文本 CRLF 归一为 LF，避免 Windows 换行造成伪版本变化。

## Windows 官方演练

先更新仓库，启动 `experiments/jammers-simulator.exe`，在界面在线登录并选择相应题型的**演练**。接口就绪后，使用界面当前案例编码运行：

```powershell
$env:JAMMERS_ROBOT_ID = '<当前登录队号>'

# 可在启动模拟器前执行；始终发送 0 条请求
python experiments/baseline_v2/run.py --backend official --mode practice --problem 4 --case-code '<界面案例编码>' --dry-run

# 真正连接本机模拟器，执行这一局演练
python experiments/baseline_v2/run.py --backend official --mode practice --problem 4 --case-code '<界面案例编码>' --name practice_q4_01

# 同一个演练入口也有 PowerShell 包装器
# 二选一执行，不能对同一局同时启动两个客户端
./experiments/baseline_v2/run_practice.ps1 -Problem 4 -CaseCode '<界面案例编码>' -Name practice_q4_02
```

Q3 将 `--problem` 改为 3。使用不同端口时加 `--base-url http://127.0.0.1:端口`。官方接口只接受本机回环地址，macOS 上真实 official 运行会直接拒绝；`--dry-run` 不受平台限制。队号只从环境变量读取，事件请求中的队号写为 `REDACTED`。

结束后，将界面总数、清除数与证据文件关联到这局，再自动重算指标：

```powershell
python experiments/baseline_v2/record_result.py --name practice_q4_01 --case-code '<同一案例编码>' --total 16 --cleared 16 --evidence 'C:/local-evidence/practice_q4_01.png'
```

上面的 16 仅为命令示例，必须替换为这局界面实际数值。证据仅记录文件名与 SHA-256，原文件仍在本机。若界面有精确的程序运行秒数，可另加 `--runtime`；不能从取整的剩余时间反推。界面没有精确值时，评价器明确标注使用服务端时间戳差值近似。总数尚未录入时，清除比例留空；录入后若未全清，验收返回失败。

`--mode formal` 只用于已经由界面启动的正式测试，并要求案例编码。客户端不能判断 UI 模式，也不启动、预约或管理正式次数；模式必须人工按 UI 正确填写。正式总数不公开，`record_result.py` 拒绝给正式局填写 `--total`。本版没有执行任何正式测试。最终仍需从模拟器导出原名加密日志；本地 JSONL 不能替代 `.jlog`，也不能据此确认上传状态。

## 模块

| 文件 | 职责与复用边界 |
| --- | --- |
| `solver/config.py` | 全部可调参数、类型检查、覆盖条件与预算校验；题给物理常数不可改 |
| `solver/geometry.py` | 复用 v1 半平面裁剪、直径、包围圆；新增退化凸包判断和旋转光学覆盖 |
| `solver/visibility.py` | 历史正观测、保证距离内负观测的朝向区间；只影响测点评分 |
| `solver/planning.py` | 候选测点、收信号位置之间的安全插值、失信号后的折半探测、几何/可接收性/移动联合评分 |
| `solver/coverage.py` | Q3 较短巡检环与逐频道整格覆盖证明；Q4 等边三角形覆盖 |
| `solver/routing.py` | 起点固定、终点自由的最近邻 + 2-opt；目标插入代价 |
| `solver/strategy.py` | 搜索、定位、清除、明确的停止证据；复用 v1 的几何更新和清除记录 |
| `solver/protocol.py` | 复用 v1 串行 HTTP、脱敏和同 ID 重试；补充完整响应校验 |
| `solver/runner.py` | 单局生命周期、退出保护、异常归档、事后真值与独立评价 |
| `solver/scenarios.py` | 与策略隔离的自建场景、固定误差规则；两版使用同一场景 |
| `solver/evaluation.py` | 直接调用 `Benchmark/evaluate.py`，增加回退、末期确认等诊断 |
| `solver/artifacts.py` | 源码清单、跨平台指纹、原子写入、Windows 文件名校验 |
| `run.py` / `run_practice.ps1` | 单局入口 / Windows 演练包装器 |
| `benchmark_suite.py` / `make_report.py` | 配对套件 / 无额外依赖的比较报告 |
| `record_result.py` / `evaluate_runs.py` / `verify_artifacts.py` | UI 结果补录 / 日志重放 / 独立几何审计 |

## 配置与验证

完整默认参数见 `solver/config.py`；`configs/default.json` 是空覆盖文件，避免两份默认参数漂移。新配置写 JSON，仅填写覆盖字段，使用 `--config` 加载。`configs/ablation_no_visibility.json`、`ablation_no_routing.json`、`ablation_no_adaptive_coverage.json` 可逐项禁用新增部件，并使用不同 prefix 运行配对消融。消融配置的存在不代表已完成该组实验。

```powershell
python -m unittest discover -s experiments/baseline_v2/tests -v
python -m unittest discover -s experiments/baseline_v1/tests -v
python -m unittest discover -s Benchmark -p 'test_*.py' -v
python experiments/baseline_v2/evaluate_runs.py experiments/baseline_v2/runs/my_paired_suite
```

测试中的 HTTP 集成使用 `127.0.0.1` 随机端口和本地测试替身，需要允许本机监听端口。它不连接官方 exe，也不使用正式次数。

默认忽略新运行目录，仅本次审核过的 `runs/validation_20260911/` 纳入 Git。每局包含完整配置、源码清单、`events.jsonl`、`decisions.jsonl`、`metrics.json`；自建离线局另有 `offline_truth.json`，策略不能从它读数据。`status=completed` 表示状态机完成；`valid_run` 表示日志计时自洽；离线验收还必须有 `all_cleared_verified=true`。预算退出、异常、未知通信结果、真值未全清均不会报告为离线验收成功。
