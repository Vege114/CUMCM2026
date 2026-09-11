# 题目转换与校对记录

- 转换日期：2026-09-11。
- 原始文件：[C题.pdf](C题.pdf)，3 页，354389 字节。
- 工具：MinerU 官方 CLI `mineru-open-api v0.5.9`，`extract` 精准模式，`vlm` 模型，语言 `ch`，输出 Markdown，默认启用表格和公式识别。
- 原始 PDF SHA-256：`2c098f6ae9dd47ae965aebdec3b9facf3de6c173f999783c1012c08fc5fb9d2d`。
- 未修改输出：[mineru-raw/C题.md](mineru-raw/C题.md)；4 张表格截图保存在 [mineru-raw/images/](mineru-raw/images/)。
- 原始 Markdown SHA-256：`b2f10d53fbbf4d11326c95f7223937dcbbeed758fa5c7233347b299a58328d9d`。
- 阅读版：[C题.md](C题.md)，保留 MinerU 生成的 HTML 表格及合并单元格。

## 校对记录

逐页核对原 PDF 的文字、数值和四张表格。阅读版仅补齐漏识别内容与调整标题格式，未改写题意。

| 原 PDF 页码 | MinerU 原始输出 | 阅读版处理 |
| --- | --- | --- |
| 1 | 缺少比赛名称和论文格式提示 | 按 PDF 补回首部两行 |
| 1 | “问题 若每天的电价相同” | 补齐为“问题 2” |
| 2 | “超出部分的电价是交易时刻电价的 倍” | 按 PDF 补齐为“1.5 倍” |
| 3 | 缺少“附件 5 结果文件夹” | 在 `result1.xlsx` 说明前补回 |
| 1–2 | 问题标题混排在正文中 | 将四问分为 Markdown 二级标题，题名改为一级标题 |

特别核对：紧急购电为交易时刻电价的 5 倍；下调部分违约电价为 50%；上调超出部分为 1.5 倍；储电范围 1200–10800 kWh；题目附件覆盖全年，结果模板要求的全年输出从 2025-02-01 开始。

## 重新转换

在本机环境中设置 `MINERU_TOKEN` 后，从仓库根目录运行：

```sh
mineru-open-api extract "C题/C题.pdf" --model vlm --language ch --format md --output ".work/mineru/c/" --timeout 900
```

核对新结果后再更新阅读版，并保留可追溯的校对记录。Token 不写入命令示例或仓库文件。
