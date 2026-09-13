# 全文决定执行与验收记录

本记录依据团队逐项人工决定执行。计算基线为459cdb2，当前派生分支为`codex/exp008-refund-coldstart`；F10/F12明确改变原冻结结果，未把旧公式配上新数字。

| 项目 | 人工决定 | 执行动作 | 证据路径 |
|---|---|---|---|
| F01 | A | 问题一正文与附录统一两层MIP，删除DP解释 | `Xelatex/sections/05-问题一.tex` |
| F02 | A | 指定槽、储能汇总及图使用同一第二层；三层诊断移历史归档 | `Xelatex/q1_run/revised.json` |
| F03 | A | Q2先原始树模型等权融合，再共同Ridge28及非递归半记忆 | `Xelatex/sections/06-问题二.tex` |
| F04 | B+备注 | 简述模式规划/固定模式修正/执行；边界集中表列 | `Xelatex/sections/06-问题二.tex` |
| F05 | A | 使用新全年连续策略评价切片替换Q2全部结果 | `data/results/exp009/q2/` |
| F06 | A，受F10覆盖 | 保留exp008情景LP控制体系并按F10修改退款，移出旧严格MIP | `experiments/exp009/q34_planner.py` |
| F07 | A+备注 | 往返90%，两方向sqrt(0.9)，不讨论另一解释 | `Xelatex/sections/03-模型假设.tex` |
| F08 | A | 题面物理参数与各问工程设置分表；Q1预算基数与费用分解 | `Xelatex/sections/03-模型假设.tex` |
| F09 | 人工备注 | 不专设争议讨论；删除错误的全场景严格保证，实际模型写LP | `Xelatex/sections/07-问题三.tex` |
| F10 | C | 新派生分支放开下调并负号退款，真实重跑Q3/Q4-3全年 | `experiments/exp009/q34_run.py` |
| F11 | B+备注 | 正文按相对午夜最终净差额一次结算，工作簿填总常规量 | `Xelatex/sections/04-符号说明.tex` |
| F12 | 人工备注 | 四策略各自从一月6000接入正式调度和因果冷启动；不使用旧预热 | `experiments/exp009/q12_forecast.py` |
| F13 | A | 逐问列当日剩余时域、库存估值、跨日连续和末日估值零 | `Xelatex/sections/06-问题二.tex` |
| F14 | A | 发布前信息与区间内测量反馈分别定义 | `Xelatex/sections/03-模型假设.tex` |
| F15 | B | 正文时标/单位约定；支撑字段字典及自动槽值核验 | `data/results/exp009/README.md` |
| F16 | A | Q3/Q4-3从0时起使用官方PV，负载使用HGB | `Xelatex/sections/07-问题三.tex` |
| F17 | A | 先b*再样本量收缩，修正推导 | `Xelatex/sections/07-问题三.tex` |
| F18 | A | 未来价格逐步揭示，4-2十维/4-3八维点预测，删除联合场景宣称 | `Xelatex/sections/08-问题四.tex` |
| F19 | A+备注 | 用短段说明共享购电和未来短缺权重近似，不自创体系名称 | `Xelatex/sections/07-问题三.tex` |
| F20 | A | 实际C与代理J分开，退款/紧急费用按真实价 | `Xelatex/sections/04-符号说明.tex` |
| F21 | A | Q4按各自完整方案比较，披露多项同时不同 | `Xelatex/sections/08-问题四.tex` |
| F22 | 不新增改进 | 未新增逐时刻经济消融；删除不属于当前策略的旧经济收益数字 | `Xelatex/sections/07-问题三.tex` |
| F23 | A | 用正确两层15组敏感性和明确范围的旧Q2固定回放，全文核验 | `Xelatex/sections/09-模型分析与检验.tex` |
| F24 | B+备注 | 保留各问定义与边界，运行强度定性解释，不宣称全面平稳 | `Xelatex/sections/09-模型分析与检验.tex` |
| F25 | B | 预测正文展示不重叠实际执行窗口，同口径比较 | `Xelatex/final_results/q3_forecast.tex` |
| F26 | A | 称开发年度因果回测，删除旧CNN种子统计/独立泛化保证 | `Xelatex/sections/03-模型假设.tex` |
| F27 | A | 新归档实际耗时、实际gap和可行率，限制最优性范围 | `data/results/exp009/q2/planning_audit.json` |
| F28 | A | 费用2位、电量4位；334/365分开；原精度算差额 | `data/results/exp009/README.md` |
| F29 | A | 公共符号仅保留实际共用量，删除未用CVaR/备用符号 | `Xelatex/sections/04-符号说明.tex` |
| F30 | A | 填摘要、评价改进推广，保留标题且限定平稳性 | `Xelatex/sections/00-摘要.tex` |
| F31 | A | 五簿集中正式目录，实际源码/依赖/导入闭包/哈希/复核入口 | `scripts/finalize_exp009.py` |
| F32 | A | 保留必要图表并补正文引导，文件表补题注，删除未引用文献 | `Xelatex/sections/12-参考文献.tex` |
| F33 | 人工备注 | 本轮不压正文页数；既有约289dpi接受；最终重新编译统计 | `Xelatex/数模通用模板.tex` |
| F34 | B | 可追溯过程说明形成2页AI详情PDF及源码，清理占位 | `Xelatex/AI工具使用详情.tex` |
| F35 | A | 更新入口说明与历史目录标记，旧隐藏快照隔离 | `Xelatex/分章节使用说明.md` |

## 适用范围与明确暂缓项

用户本轮要求落实既有决定、参考math-modeling文风并编译现有TeX，故保留现有项目结构，不重建题目分析、候选模型或Word版本。已实际读取Skill根入口、编程手、论文手、写作规范、Subagent调度、截止协议、LaTeX、Excel与绘图入口。稀疏安装缺少的子文件从原Skill仓库按需读回，暂存在忽略目录。

F33明确暂不考虑页数限制并接受约289dpi旧图，本轮保留该安排，后续按最终编译页数优化；当届官方正文上限30页并未因此改变。官方出处：[2026格式规范](https://www.mcm.edu.cn/html_cn/node/4cd596519c9eb9fbd866398f6df0caa3.html)。竞赛截止2026年9月13日20:00（北京时间），参见[主办方通知](https://www.csiam.org.cn/upload/shuxue/69c3870950b04.pdf)。

## 验收状态

2026年9月13日完成。P1、P2、W1、W2独立验收全部PASS，详见[独立验收总表](audit/gate_summary.json)。四个年度方案均完成365日连续求解，五份工作簿完成独立逐数值单元核对，共计313981格。全文35项人工决定均已落实；F33的页数压缩按用户决定继续暂缓。

- 问题一：两层正式轨迹独立重求一致；第二层费用34115.03元。
- 问题二、三、四无更新、四有更新：334日费用分别为13195653.64、12788960.26、14008609.54、13539823.62元。全年365日与一月费用另列于[修订结果报告](report.md)。
- 源文件闭包48个，原文件与论文程序副本SHA一致。另核对Git索引中164项源码、输入、轨迹和工作簿，全部与清单SHA一致；原CSV仅固定字节保存方式，数据值未变。
- Q3/Q4-3由独立审阅者完整重跑365日，全部数组及NPZ哈希一致。Q2/Q4-2按5秒MIP预算运行，全年数值复核及代表日重解通过，不承诺跨硬件轨迹逐位相同。
- 支撑包18983068字节（18.10 MiB），完整365日数据全部保留。重复切片由准备脚本恢复并逐SHA核对；隔离解压后的准备、核验和五类求解入口全部退出0。
- 全文XeLaTeX构建退出0，警告0；validate退出0，无未定义引用、缺字、空白页或未嵌入字体。153页中摘要1页，正文及参考部分47页（第2—48页），附录105页。
- 正文内容计22754单位，48个编号公式、10幅图、50张表、4条引用文献；四问均有图。最低原图DPI为289，按F33接受。
- 独立扫描153页全部文字边界，并实际查看12个渲染页；AI详情PDF两页也逐页查看，均PASS。主线程另外查看摘要、文件列表和程序附录入口。
- 全仓TeX编译产物与中间文件已加入忽略规则，并从索引移除原六份编译PDF；原始题目与绘图PDF正常保留。对源码保留原哈希对应的末尾空行，对SVG保留自动导出格式；排除这两类非语义格式并识别CRLF后，Git差异格式检查通过。

### 实际执行的关键入口

| 命令/步骤 | 退出码 | 证据 |
|---|---:|---|
| Q1正式两层与独立附录求解 | 0 | `audit/q1_reproduction.json` |
| Q12四季全年：`python -m experiments.exp009.q12_run --scenario 2/4-2 --days 365`（分两进程分别执行） | 0 | 各问`protocol.json`、`summary.json`、`planning_audit.json` |
| Q34全年：`python -m experiments.exp009.q34_run --scenario 3/4-3 --days 365`（分别执行） | 0 | 各问`completion.json`、`verification_365.json` |
| `python scripts/finalize_exp009.py` | 0 | `复现清单.json`与本地ZIP |
| `python scripts/prepare_exp009_support.py` | 0 | `audit/package_smoke.json` |
| `python scripts/verify_exp009_delivery.py` | 0 | `data/results/exp009/delivery_verification.json` |
| LaTeX Skill：`latex_paper.py doctor --engine xelatex --bibliography-backend none` | 0 | 实际环境检查；XeLaTeX/latexmk/Poppler/pypdf可用 |
| LaTeX Skill：`latex_paper.py build Xelatex/数模通用模板.tex --engine xelatex --timeout 180 --publish 完整论文.pdf` | 0 | `latex_build.json`、PDF build manifest |
| LaTeX Skill：`latex_paper.py validate ... --contest cumcm --quality-checks --questions q1 q2 q3 q4 --body-start-page 2 --appendix-start-page 49 --min-image-dpi 289` | 0 | `latex_validate.json`；F33原因已记录 |

前两次全文构建发现公式行宽、文件列表行宽与等宽字体粗体缺项，已实际修复；最终构建零警告，没有通过忽略警告掩盖问题。已使用可用的系统TeXLive 2026/XeLaTeX，不涉及新装编译器。

PDF SHA-256：`0a89e9e24dcd8e0be7ee4c03d903fcb166ba70efb10429fb0c80a25a7f71ea57`。支撑包 SHA-256：`aec168e3383941c27f495baba4f68765cc411053f99118793d00dc0099ab8694`。PDF与ZIP按用户要求保存在本地且被Git忽略；源码、五份正式工作簿、可复核数据和验收记录提交远程分支。
