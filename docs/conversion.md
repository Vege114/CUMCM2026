# 文档转换与校对记录

转换日期：2026-09-10。工具：MinerU Document Extractor / mineru-open-api v0.5.9。三份正式文档均通过 MinerU extract 转为 Markdown，附件采用默认模型；题目另用 `--model vlm` 重提取，缺字仍存在，随后对照原 PDF 全部四页校对。

## 转换范围

| 原始文件 | 阅读版 | 结果 |
| --- | --- | --- |
| `B题/B题.pdf` | `docs/markdown/B题.md` | 完成，校对缺字 |
| `B题/附件/附件1.docx` | `docs/markdown/附件1.md` | 完成，保留转换排版 |
| `B题/附件/附件2.docx` | `docs/markdown/附件2.md` | 完成，保留转换排版 |

`~$附件1.docx`、`~$附件2.docx` 是 Word 临时锁文件，排除转换及提交。模拟器是可执行文件，不作文本转换。仓库外的 `CUMCM2026B/` 不属于本次指定的 B题目录。

图片保存在 `docs/markdown/images/`，与 Markdown 一并提交。附件中的 HTML 表格是 Markdown 支持的内嵌格式。附件 2 的代码示例保留了转换产生的 HTML 强调标签和下划线转义，阅读时请以原 DOCX 为准，不直接复制为可运行程序。

## 题目校对记录

`docs/mineru-raw/B题.md` 保留未修改的 MinerU 提取内容；其配图同时保存在同级 `images/` 中。阅读版 `docs/markdown/B题.md` 做了如下修正，均依据原 PDF 页面图像：

- 第 1 页：恢复竞赛标题和阅读格式规范提示；恢复 x/y 轴、问题 2 编号、初始频道 1、问题 3 对附录 3 的引用。
- 第 2 页：问题 4 的“其他条件”恢复为问题 3；方位角参考轴恢复为 x；恢复图 1 引用及图注 G、S。
- 第 3 页：交会图说明恢复 G；恢复条目 (8)、光学定位距离 20 米、定位耗时 3 秒、激光清除耗时 2 秒。
- 第 4 页：核对测试窗口、运行时限、截止时间与附录 4，未作语义修改。

不对题目作解答，原始 PDF、DOCX 和模拟器均保持原样。公式、代码和正式要求仍以原件为准。

## 原始文件 SHA-256

```text
81C992A9BEE5376308C4768B58719B177F17D6EE1F9861577A851924C5A838CA  B题/B题.pdf
20A27603EA81EFA8F11658B3FA5B859A69AA3FF88FB3664D2ABB80D740B47553  B题/附件/附件1.docx
C882513D5B7E0EC50F3068570EA55FDC1B5C4A5FA2D6E9E54E79B33CF0858CB2  B题/附件/附件2.docx
2373B9E7AF83735A04309E2983EB433EC46FAF7E0B8494410CE7FDED2A297C27  jammers-simulator.exe
```

## 重新转换

先在当前进程环境中设置 `MINERU_TOKEN`，再从根目录运行 `scripts/convert-docs.ps1`。默认输出至被忽略的 `.work/mineru-refresh/`，人工核对差异后更新阅读版及本记录。脚本不保存凭证，且不会覆盖已校对的阅读版。
