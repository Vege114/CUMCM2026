# CUMCM 2026 · C 题

**微网与外部电网电力调控策略**。当前已完成题目转换、原始附件整理和仓库初始化，尚未开展建模实验。

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
paper/                     LaTeX 写作区，仅 .gitkeep 占位
data/                      数据区
  raw/                     附件 1–4 原始数据
  templates/               附件 5 的五个原始结果模板
  processed/               清洗、时间对齐后的派生数据
  results/                 经核验的答题工作簿、论文表格及图片
  manifest.json            10 个原始文件的路径、大小、SHA-256 和工作表名称
```

## 开始工作

1. 阅读 [题目 Markdown](C题/C题.md)，必要时对照 [原始 PDF](C题/C题.pdf) 与 [校对记录](C题/conversion.md)。
2. 按 [答题交付清单](C题/requirements.md) 确认四问的输入、时间范围和结果文件。
3. 依据 [数据区说明](data/README.md) 读取附件，在 [实验区](experiments/README.md) 开展建模与验证。
4. 将核验后的结果整理到 `data/results/`；论文模板后续由小组加入 `paper/`。

## 协作约定

- 原始 PDF、`data/raw/` 和 `data/templates/` 保持原始字节；清洗数据与填好的结果另存。
- 实验记录写明输入、假设、参数、随机种子（如使用）、依赖版本、运行命令和 Git 提交，论文引用可复现的结果。
- 区分功率 kW、电量 kWh、电价元/kWh 和费用元；功率转区间电量时明确时间间隔及插值或积分方式。
- 涉及按时刻制定策略时，记录决策时可用信息，避免使用尚未发布的预报或未来实测数据。
- Token、密钥和本机配置保存在仓库外或忽略文件中。

本次从 B 题切换到 C 题，当前工作区的 B 题题目、代码、报告和模拟器已清空；既有 Git 历史保留。
