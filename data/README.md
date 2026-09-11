# 数据区

## 原始附件

以下文件仅从原 `C题/附件/` 移动，未修改内容。文件大小、SHA-256 和工作表名记录在 [manifest.json](manifest.json)，包括原题 PDF 在内共 10 个原始文件。

| 文件 | 内容 | 主要用途 |
| --- | --- | --- |
| [raw/附件1.xlsx](raw/附件1.xlsx) | 某天电价、小区负载、光伏预测功率 | 问题 1；问题 2、3 的固定电价 |
| [raw/附件2.xlsx](raw/附件2.xlsx) | 2025 年小区负载与光伏实际功率，两个工作表 | 问题 2–4 |
| [raw/附件3.xlsx](raw/附件3.xlsx) | 2025 年每日四次发布的未来 24 小时整点光伏预报 | 问题 3、4 |
| [raw/附件4.xlsx](raw/附件4.xlsx) | 2025 年不同时间的电价 | 问题 4 |

## 原始结果模板（附件 5）

| 模板 | 工作表 |
| --- | --- |
| [templates/result1.xlsx](templates/result1.xlsx) | 计划购电量、充放电量 |
| [templates/result2.xlsx](templates/result2.xlsx) | 计划购电量、充放电量、紧急购电量 |
| [templates/result3.xlsx](templates/result3.xlsx) | 计划购电量、调整购电量、充放电量、紧急购电量 |
| [templates/result4-2.xlsx](templates/result4-2.xlsx) | 计划购电量、充放电量、紧急购电量 |
| [templates/result4-3.xlsx](templates/result4-3.xlsx) | 计划购电量、调整购电量、充放电量、紧急购电量 |

`templates/` 内均为原始模板，尚未填写计算结果。

## 派生数据与交付结果

- `processed/`：清洗与时间对齐的中间数据。默认忽略可再生成文件，处理逻辑留在实验区；需要保留小型基准数据时，明确调整忽略规则。
- `results/`：经核验的五个结果工作簿，以及论文使用的汇总表、图和结果说明。此目录纳入 Git，不存放临时运行缓存。
- 临时运行产物放在 `experiments/problemN/<实验名>/runs/`，默认忽略。核验后选择需要交付的产物放入 `results/`。

数据处理时应记录原始时间含义、区间边界、单位、缺失值处理以及插值或积分方法。时间标签 `0:00+1` 与预报发布时刻、预报目标时刻应分别处理。
