# CUMCM 2026 · B 题

**无线电干扰源的快速自动定位与清除**。已完成模块化基础模型：官方模拟器问题 3、4 各一次演练全清，并建立可复用的 Benchmark。

当前入口：[改进模型 v2 与 Windows 运行方式](experiments/baseline_v2/README.md) · [v2 离线验证报告](experiments/baseline_v2/REPORT.md) · [v1 官方演练报告与图表](experiments/baseline_v1/REPORT.md) · [题目指标 Benchmark](Benchmark/README.md)。v2 已在 macOS 完成代码和配对离线验证，尚待 Windows 官方模拟器演练。v1 问题 3 演练清除 13/13、平均 380.40 虚拟秒/个；问题 4 演练清除 16/16、平均 1103.17 虚拟秒/个。均为演练结果，尚未使用正式测试机会。

## 目录

| 路径 | 用途 |
| --- | --- |
| `B题/` | 比赛原始 PDF 和 DOCX，保持原样 |
| `docs/markdown/` | MinerU 转换后的题目、附件和配图；题目已按原 PDF 校对缺字 |
| `docs/mineru-raw/` | 题目未经校对的 MinerU 原始输出 |
| `docs/problem-map.md` | 四问的交付物与目录对应关系，仅整理要求 |
| `docs/conversion.md` | 转换方法、文件校验值与校对记录 |
| `paper/` | 论文目录预留，LaTeX 模板由小组后续自行添加 |
| `src/` | 通信、公共组件以及问题 1–4 的待实现目录 |
| `config/` | 不含账号的示例配置 |
| `data/` | 后续公开输入及整理数据的目录约定 |
| `experiments/baseline_v1/` | 基础模型、配置、测试、每局产物与可视化报告集中管理 |
| `experiments/baseline_v2/` | 定向可接收性、覆盖与路径改进、Windows 入口、分层配对离线验证 |
| `Benchmark/` | 题目指标字典、场景规范、独立日志评价器 |
| `results/` | 后续经核验、可供论文引用的结果 |
| `submission/` | 问题 3、4 正式日志与最终支撑材料 |
| `scripts/` | 文档转换脚本 |
| `tests/` | 后续单元测试和离线协议测试 |
| `experiments/jammers-simulator.exe` | 用户移入实验目录的 Windows 模拟器，二进制保持不变 |
| `tools/README.md` | 模拟器使用和文件管理说明 |

## 阅读顺序

1. [题目](docs/markdown/B题.md)
2. [附件 1：模拟器使用说明](docs/markdown/附件1.md)
3. [附件 2：通信接口说明及编程指南](docs/markdown/附件2.md)
4. [任务目录映射](docs/problem-map.md)

原始文档是依据；附件转换保留了 HTML 表格和原有代码排版，代码示例不能直接作为源文件运行。

## 论文环境

小组计划使用 LaTeX 写作，模板由小组后续自行添加到 `paper/`。当前仅预留目录，编译方式待模板加入后补充。

## 协作约定

- 原题、原始附件与模拟器不作修改；转换校对记录写入 `docs/conversion.md`。
- 当前基础版使用 Python 标准库；通信、几何、策略、规划、配置和离线环境在 `experiments/baseline_v1/baseline/` 分模块管理，便于整体复制和逐版比较。
- 结果记录关联 Git 提交、配置和日志；论文只引用已核验结果。
- 问题 3 和问题 4 各预留三次正式测试日志，保留模拟器导出文件名及原始内容。
- token、登录状态、参赛队号配置保存在本机环境或被忽略的本地配置中。
- 已使用公开 HTTP 接口完成 Q3/Q4 演练；账号和模拟器内部数据不进入代码，正式测试机会未消耗。
