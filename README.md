# CUMCM 2026 · B 题

**无线电干扰源的快速自动定位与清除**。本仓库当前仅完成资料转换与项目初始化，尚未开展建模、算法实现或模拟器测试。

## 目录

| 路径 | 用途 |
| --- | --- |
| `B题/` | 比赛原始 PDF 和 DOCX，保持原样 |
| `docs/markdown/` | MinerU 转换后的题目、附件和配图；题目已按原 PDF 校对缺字 |
| `docs/mineru-raw/` | 题目未经校对的 MinerU 原始输出 |
| `docs/problem-map.md` | 四问的交付物与目录对应关系，仅整理要求 |
| `docs/conversion.md` | 转换方法、文件校验值与校对记录 |
| `paper/` | XeLaTeX 中文论文模板，按章节分文件 |
| `src/` | 通信、公共组件以及问题 1–4 的待实现目录 |
| `config/` | 不含账号的示例配置 |
| `data/` | 后续公开输入及整理数据的目录约定 |
| `experiments/` | 演练记录模板；本地运行输出置于 `runs/` |
| `results/` | 后续经核验、可供论文引用的结果 |
| `submission/` | 问题 3、4 正式日志与最终支撑材料 |
| `scripts/` | 文档转换与论文编译脚本 |
| `tests/` | 后续单元测试和离线协议测试 |
| `jammers-simulator.exe` | 用户提供的 Windows 模拟器，保留原位置 |
| `tools/README.md` | 模拟器使用和文件管理说明 |

## 阅读顺序

1. [题目](docs/markdown/B题.md)
2. [附件 1：模拟器使用说明](docs/markdown/附件1.md)
3. [附件 2：通信接口说明及编程指南](docs/markdown/附件2.md)
4. [任务目录映射](docs/problem-map.md)

原始文档是依据；附件转换保留了 HTML 表格和原有代码排版，代码示例不能直接作为源文件运行。

## 编译论文

安装含中文支持的 TeX Live，使用 XeLaTeX + latexmk。从仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-paper.ps1
```

输出为 `paper/build/main.pdf`。也可执行 `latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error paper/main.tex`。
入口为 `paper/main.tex`，正文位于 `paper/sections/`，图片位于 `paper/figures/`，表格位于 `paper/tables/`。
当前为小组写作骨架，并非已核验的官方提交模板；提交前须对照比赛论文格式规范调整封面、页码和匿名信息。

## 协作约定

- 原题、原始附件与模拟器不作修改；转换校对记录写入 `docs/conversion.md`。
- 建模语言尚未限定；代码按问题分目录，通信实现独立放在 `src/simulator/`。
- 结果记录关联 Git 提交、配置和日志；论文只引用已核验结果。
- 问题 3 和问题 4 各预留三次正式测试日志，保留模拟器导出文件名及原始内容。
- token、登录状态、参赛队号配置保存在本机环境或被忽略的本地配置中。
- 本次初始化没有启动模拟器，没有调用 `/enter`，也没有消耗正式测试机会。
