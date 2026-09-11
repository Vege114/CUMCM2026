# CUMCM 2026 · C 题

**微网与外部电网电力调控策略**。当前已完成题目转换、原始附件整理及两阶段方案统一，尚未实现调度求解或验证数值收益。

## 当前统一方案

以队友最新交接的第四轮决策为准，先读[团队统一建模口径](paper/团队统一建模口径.md)，再读[第一阶段分析](paper/第一阶段_赛题深度结构化分析.md)和[第二阶段模型](paper/第二阶段_各小问模型构建与创新设计.md)。旧稿在paper/archive，仅作追溯。四问主线为：三层词典序调度→滚动周期预测与备用→提前量感知多阶段风险调度→滚动电价预测。

## Python 环境

既有环境配置为 **Python 3.12 + TensorFlow 2.21 / `tf.keras`**；已安装依赖不代表模型路线，当前主方案不要求强化学习。若后续另做强化学习对照，则使用 **Gymnasium** 定义环境、状态、动作与交互接口，策略网络及训练代码使用 TensorFlow。Gymnasium 本身不提供 PPO、DQN 等训练算法，具体算法随建模实验实现。

依赖在 [pyproject.toml](pyproject.toml) 中声明，精确版本及包校验信息由 `uv.lock` 锁定；[uv](https://docs.astral.sh/uv/getting-started/installation/) 管理仓库根目录的 `.venv/`。`.venv/`、`venv/`、`env/` 和 `.cache/` 已加入 `.gitignore`，不提交环境和下载缓存。

在仓库根目录执行：

```bash
# 首次安装或同步队友提交的依赖（自动使用 .python-version 指定的 Python）
uv sync --locked

# 验证 TF 训练/模型保存、Gymnasium 交互、Excel 读写和绘图
uv run --locked python scripts/check_environment.py

# 启动 Notebook；该命令使用项目 .venv 中的 Python
uv run --locked jupyter lab

# 查看各实验 runs/ 内的 TensorBoard 日志
uv run --locked tensorboard --logdir experiments
```

命令行运行脚本可用 `uv run --locked python <脚本路径>`。也可先激活环境（macOS/Linux：`source .venv/bin/activate`；Windows PowerShell：`.venv\Scripts\Activate.ps1`）；编辑器或 Notebook 的解释器选择项目 `.venv` 中的 Python。退出环境执行 `deactivate`。

| 用途 | 依赖 |
| --- | --- |
| 神经网络、预测与强化学习策略训练 | TensorFlow / `tf.keras`、TensorBoard |
| 强化学习环境接口和经典控制示例 | Gymnasium（含 classic-control） |
| 数值计算、数据清洗、预处理与评估指标 | NumPy、SciPy、pandas、scikit-learn |
| 附件读取与结果工作簿导出 | openpyxl |
| 可视化、配置与进度显示 | Matplotlib、Seaborn、PyYAML、tqdm |
| 交互实验与代码检查（默认安装的 dev 组） | JupyterLab、ipykernel、Ruff |

新增依赖用 `uv add <包名>`，开发工具用 `uv add --dev <包名>`，同步提交 `pyproject.toml` 和 `uv.lock`。只运行脚本、无需 Notebook/代码检查工具时，可以用 `uv sync --locked --no-dev`。

本机为 Apple Silicon，默认安装 CPU 版 TensorFlow；当前配置不包含 Metal 或 CUDA 加速插件。Python 3.12 在 [TensorFlow 官方支持列表](https://www.tensorflow.org/install/pip) 内。本环境先保证训练与依赖可复现，GPU 环境按实际训练设备另行配置。运行检查只产生临时文件和忽略的工具缓存，不改动原始附件或生成建模结果。

## 目录

```text
C题/                       题目资料
  C题.pdf                  原始题目
  C题.md                   MinerU 转换并按 PDF 校对的题目
  conversion.md            转换方法与校对记录
  requirements.md          四问任务及答题交付清单
  mineru-raw/               未修改的 MinerU 输出及表格截图
experiments/               实验区
  common/                  数据读取、调度约束、计费、评估等公共代码预留
  problem1/                问题 1：单日计划购电
  problem2/                问题 2：全年计划与紧急购电
  problem3/                问题 3：引入日内预报的调整策略
  problem4/                问题 4：波动电价下复算问题 2、3
  record-template.md       实验记录模板
paper/                     统一口径、两阶段分析、新版流程图及旧稿归档
data/                      数据区
  raw/                     附件 1–4 原始数据
  templates/               附件 5 的五个原始结果模板
  processed/               清洗、时间对齐后的派生数据
  results/                 经核验的答题工作簿、论文表格及图片
  manifest.json            10 个原始文件的路径、大小、SHA-256 和工作表名称
scripts/check_environment.py 环境自检
pyproject.toml             项目依赖与工具配置
uv.lock                    可复现的依赖锁文件
```

## 开始工作

1. 阅读 [题目 Markdown](C题/C题.md)，必要时对照 [原始 PDF](C题/C题.pdf) 与 [校对记录](C题/conversion.md)。
2. 按 [答题交付清单](C题/requirements.md) 确认四问的输入、时间范围和结果文件。
3. 按上文安装并验证 Python 环境，依据 [数据区说明](data/README.md) 读取附件，在 [实验区](experiments/README.md) 开展建模与验证。
4. 将核验后的结果整理到 `data/results/`；论文模板后续由小组加入 `paper/`。

## 协作约定

本轮问题 2—4 的 GPU 神经网络预测与基础调度见 [实验报告索引](reports/latest.md)，可复用的八部分报告模板和历史对比工具见 [报告目录说明](reports/README.md)。`main` 同步保存完整报告包；训练代码、GPU 依赖和四个结果工作簿保存在 [codex/neural-forecasting-v1 实验分支](https://github.com/Vege114/CUMCM2026/tree/codex/neural-forecasting-v1)。

- 原始 PDF、`data/raw/` 和 `data/templates/` 保持原始字节；清洗数据与填好的结果另存。
- 实验记录写明输入、假设、参数、随机种子（如使用）、依赖版本、运行命令和 Git 提交，论文引用可复现的结果。
- 区分功率 kW、电量 kWh、电价元/kWh 和费用元；功率转区间电量时明确时间间隔及插值或积分方式。
- 涉及按时刻制定策略时，记录决策时可用信息，避免使用尚未发布的预报或未来实测数据。
- Token、密钥和本机配置保存在仓库外或忽略文件中。

本次从 B 题切换到 C 题，当前工作区的 B 题题目、代码、报告和模拟器已清空；既有 Git 历史保留。
