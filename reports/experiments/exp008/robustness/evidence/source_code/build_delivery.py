"""Package measured tables, figure captions, and a paper-writing companion."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'data/results/exp008/robustness'
FIG = ROOT/'paper/figures/exp008_robustness'
OUT = ROOT/'reports/experiments/exp008/robustness'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=OUT/'evidence'
    for section in ('q1','forecast','temporal','execution'):
        for path in (DATA/section).iterdir():
            if path.is_file() and path.suffix in ('.csv','.json','.md','.py'):
                dest=evidence/section/path.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest)
    source_dir=evidence/'source_code'
    source_dir.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source_dir/path.name)
    figures=json.loads((FIG/'figure_manifest.json').read_text())
    sensitivity=pd.read_csv(DATA/'forecast/sensitivity_annual.csv')
    net=sensitivity[sensitivity.channel=='net']
    pooled=pd.read_csv(DATA/'forecast/cv_pooled_metrics.csv')
    pooled=pooled[(pooled.channel=='net')&(pooled.test_days_per_fold==28)]
    windows=pd.read_csv(DATA/'temporal/window_metrics.csv')
    windows['scenario']=windows.scenario.astype(str)
    q2=windows[(windows.scenario=='2')&(windows.strategy=='exp008_final')]
    annual=q2[q2.window_type=='full'].iloc[0]
    months=q2[q2.window_type=='month']
    tails=q2[q2.window_type=='chronological_tail'].sort_values('requested_tail_fraction')
    bootstrap=pd.read_csv(DATA/'temporal/bootstrap_summary.csv')
    b14=bootstrap[(bootstrap.method=='paired_moving_block')&(bootstrap.block_days==14)].iloc[0]
    execution=pd.read_csv(DATA/'execution/stress_metrics.csv')
    noise=pd.read_csv(DATA/'execution/noise_summary.csv')
    nominal=net[net.name=='baseline'].iloc[0]
    hyper=pd.read_csv(DATA/'forecast/hyperparameter_metrics.csv')
    hn=hyper[hyper.channel=='net']
    frame=pd.DataFrame({'验证窗口/日':pooled.validation_days.astype(int),
        '3个起点合并净负荷RMSE/kW':pooled.rmse_kw.map(lambda v:f'{v:.3f}'),
        '日累计净能量RMSE/kWh':pooled.daily_energy_rmse_kwh.map(lambda v:f'{v:.2f}'),
        '合计测试日数':pooled.days.astype(int)})
    blocks=[]
    for row in figures:
        links=' · '.join(f'[{ext.upper()}](../../../../{path})' for ext,path in row['outputs'].items())
        blocks.append(f"| {row['id'][:2]} | {row['title']} | {links} |")
    figure_table='| 图号 | 内容 | 文件 |\n| --- | --- | --- |\n'+'\n'.join(blocks)
    bias=lambda family,level:execution[(execution.family==family)&np.isclose(execution.level,level)].iloc[0].cost_change_pct
    report=f'''# 第八次实验：敏感性与鲁棒性分析

本分析围绕已经选定的 exp008 最终成果展开，保留原模型与主结果。第二问正式费用仍为 **{annual.total_cost_yuan:,.2f} 元**，相对 exp006 下降 **{annual.cost_reduction_pct:.4f}%**。新增分析用于说明参数、时间划分和扰动条件下的稳定性，不将敏感性表中的较优点重新替换为论文主模型。

**结论：预测参数与跨时段表现较稳定；持续低估负荷、持续高估光伏会使既定购电计划产生明显的紧急购电成本。** 这一经济脆弱点应与稳定结果一并报告。

[全部图像预览](../../../../paper/figures/exp008_robustness/overview.png) · [15页图集PDF](../../../../paper/figures/exp008_robustness/all_figures.pdf) · [LaTeX插图片段](figures.tex) · [完整图注](captions.md) · [核验记录](verification.json)

## 1. 本次实际完成的分析

| 分析 | 范围 | 实际执行 | 可支持的结论 |
| --- | --- | --- | --- |
| 预测融合与后处理 | 18个设置×334日 | 逐日重算因果Ridge与负荷记忆 | 固定原始预测下的局部参数稳定性 |
| 训练/验证划分 | 3个起点×3种验证长度 | 36个模型真实重训，7/14/28日嵌套评分 | 开发期时间顺序回测稳定性 |
| 核心树参数 | 4种扰动×3个起点 | 另有24个模型真实重训 | 学习率、叶节点样本数邻域稳定性 |
| 第一问调度参数 | 15个设置 | 每组均重求两阶段MILP | 附件1典型日参数敏感性 |
| 时间窗口与压力分组 | 四个年度最终方案 | 独立从原始数据核验并重评分 | 月份、窗口、压力日的已实现表现 |
| 配对区块重采样 | 2种方法×3种块长×4096次 | 24576次配对抽样 | 既定策略的条件性描述区间 |
| 固定计划执行压力 | 37条确定性设置记录+120次噪声回放 | 各次连续执行334日 | 固定购电与模式计划的执行暴露 |

37条确定性记录包含不同参数组重复的中心点，并非37个独立实验；噪声试验共用随机数，也不是相互独立的数据样本。问题三、4-2、4-3保留自身最终预测与信息规则；本次详细预测参数重训针对问题二，不把结果泛化成其他三问已全部重训。

## 2. 预测参数敏感性（图01、图15）

第二问基准为原始HGB/ExtraTrees各0.5融合，经过共同Ridge28，再加0.5倍昨日已完成的原始Ridge日均负荷残差。中心方案的96,192个负荷/光伏预测值逐元素复现冻结档案，净负荷RMSE为 **{nominal.rmse_kw:.3f} kW**。

融合HGB权重取0.3/0.4/0.5/0.6/0.7，校准窗口取14/21/28/35/42日，正则倍率取0.5/0.75/1/1.25/1.5/2，昨日记忆系数取0.25/0.4/0.5/0.6/0.75。每次只调整一个参数族；窗口按原模型族规则同步取半衰期=窗口/2，不能单独归因为历史长度。

全部18个设置的净负荷RMSE变化为 **{net.rmse_kw_change_pct.min():+.3f}% 至 {net.rmse_kw_change_pct.max():+.3f}%**；日累计净能量RMSE的最大增幅为 **{net.daily_energy_rmse_kwh_change_pct.max():.3f}%**。这支持当前预测链对所测邻域参数具有稳定性，但未证明全局最优或任意参数下均稳定。

追加真实重训将HGB学习率0.08调整为0.064/0.096，将ExtraTrees叶节点最少样本10调整为8/12；每次只重训被改变的模型族，其余保持冻结来源。三个起点的净负荷RMSE变化为 **{hn.rmse_kw_change_pct.min():+.3f}% 至 {hn.rmse_kw_change_pct.max():+.3f}%**。未重新运行这些预测对应的年度优化，因此不能直接把误差改善解释为调度降费。

## 3. 训练、验证与测试窗口调整（图02、图03）

取2025年4月1日、7月1日、10月1日为起点，训练均从1月8日开始，训练末日随验证长度向前移动，验证取起点前7/14/21个完整日。训练、验证、评分严格按时间先后分离。HGB用时间验证集早停；ExtraTrees的验证值仅作诊断，未用测试结果选择参数。

每个起点重训后，接下来28日**逐日午夜发布次日144槽预测**，日期推进后允许使用已经结束日期的观测更新历史特征、Ridge与记忆；不是在起点一次性预测未来28日。起点前沿用共同已发布预测作为历史暖启动，各方案从起点开始分叉。7日验证的重训输出与冻结基准最大差仅1.82×10⁻¹² kW。

{md_table(frame)}

7日验证下三个28日窗口的净负荷RMSE约为369.74、440.38、319.05 kW；改变划分带来的差异小于起点之间的季节差异。7/14/28日是同一条预测序列的嵌套评分前缀，不能视作独立重复实验；较长评分窗误差下降也不代表“预测步长越长越准确”。合并RMSE均从全部误差平方和计算，没有平均分组RMSE。

**2025年数据此前已被反复用于开发与模型选择，重划日期不会使它重新成为未使用过的独立测试集。** 论文宜使用“开发期时间顺序回测”“滚动验证”，外部泛化仍需新年度或新数据。

## 4. 第一问运行参数（图04、图05）

15组两阶段求解全部成功，第二阶段gap为0，物理违例为0；中心轨迹、费用和指标与原成果完全一致。每个设置都以该设置自己的第一阶段经济最优值计算第二阶段费用预算。容量变化保持SOC比例，效率按往返效率解释。

| 改动 | 相对最终方案费用变化 | 解释 |
| --- | --- | --- |
| δ从0.001变为0.0009/0.0011 | 约−0.010% / +0.010% | 综合平稳目标S变化+2.306% / −0.580% |
| δ减半为0.0005 | −0.050% | S恶化20.928%，充放电启动从4增至6 |
| 爬坡上限900/1100 kW每10min | +0.113% / −0.087% | 费用变化较小 |
| 同模式最短关闭10/30 min | 数值容差内不变 | 此典型日对该邻域设置不敏感 |
| 往返效率0.855/0.945 | +1.949% / −1.838% | 效率损失影响实际购电量 |
| 容量10800/13200 kWh | +2.282% / −2.077% | 容量影响可平移电量 |

TV为包含跨午夜差分的循环日净功率总变差；S还包含归一化启动与短段项，不能互相代称。这里的“启动次数”是充/放模式起段总数，不是启动与停止动作数之和。以上结果只针对附件1给定单日。

## 5. 时间与样本组成鲁棒性（图06—图10）

Q2的11个月均优于同月exp006，月度降费范围 **{months.cost_reduction_pct.min():.3f}%—{months.cost_reduction_pct.max():.3f}%**；30/60/90日所有滚动窗口均保持正改善。取全期末尾20%/30%/40%的67/101/134日，降费分别为 **{tails.iloc[0].cost_reduction_pct:.3f}% / {tails.iloc[1].cost_reduction_pct:.3f}% / {tails.iloc[2].cost_reduction_pct:.3f}%**。这些窗口只重评分冻结轨迹，窗口起点不重置SOC，不冒充重新训练。

对两策略同日费用进行配对移动块重采样，取7/14/28日块；另做季度分层版以保持季节构成。14日普通移动块的95%描述区间为 **[{b14.percentile_95_lower_pct:.3f}%, {b14.percentile_95_upper_pct:.3f}%]**，所有六种配置的区间下端均为正。这反映固定策略在现有日期组成下的稳定性，不涵盖训练随机性、反复选模偏差或新年度误差，不能写作无条件泛化保证。

真实高负荷、低光伏、高净负荷日分别按10%/20%/30%尾部阈值分组，比较同一批日期的费用。20%高负荷日和低光伏日分别改善6.614%和4.405%。分组仅作事后解释，没有把未来所属压力组传入控制器；“高负荷且低光伏”20%交集仅1日，不作普遍性推断。

图09给出问题二、三、四问两个方案的月内日均费用与紧急购电。各问使用自身真实信息边界和电价，特别是Q4价格不同，因此不能按四条费用曲线高低直接排列算法优劣。

## 6. 计费、硬件与输入扰动（图11—图14）

图11固定已执行动作，对紧急购电倍率4—6、上调倍率1.2—1.8重新结算；这是计费暴露分析，不重新优化。下调量为0，所以本档案下调倍率变化的费用影响为0，不能外推为该规则总是无关。

图12—14保持Q2的已发布购电量及模式序列，改变实况或硬件，再由物理反馈重新执行334日，每种情况连续传递自己的SOC。由于历史扰动后未来计划仍沿用原档案，这是**固定计划的条件压力回放**，不是整套模型重新训练、重新规划后的长期性能。

实况负荷持续+2%时费用上升 **{bias('load_bias',.02):.2f}%**，光伏持续−2%时上升 **{bias('pv_bias',-.02):.2f}%**；二者同时出现时上升 **{bias('combined_bias',.02):.2f}%**。联合+10%压力（负荷×1.1、光伏×0.9）使费用上升 **{bias('combined_bias',.1):.2f}%**，说明持续系统性偏差需要重点关注。

将容量减至90%、功率减至90%、往返效率降为85%时，固定计划执行费用分别上升 **{bias('capacity',.9):.2f}% / {bias('power',.9):.2f}% / {bias('efficiency',.85):.2f}%**。容量变化时初始SOC按同一比例变化，因此这是硬件与匹配初态共同改变的情景。

随机压力使用明确假设的乘性对数正态噪声：高斯对数扰动的槽间AR(1)系数为0.95、负荷/PV两通道扰动的相关系数为−0.3，将乘性因子均值校正为1并保留夜间零光伏；对数标准差1%/2%/5%/10%各30次，共用随机数。平均费用增幅分别为 **{' / '.join(f'{v:.2f}%' for v in noise.mean_cost_change_pct)}**。这些相关系数属于假设扰动，不是扰动后实况本身的相关系数。箱线与分位数是这30次合成扰动的经验分布，不是从真实天气估计的置信区间。

全部回放均通过SOC、功率、能量平衡、充放互斥及禁止紧急购电充电的核验。模型允许无限紧急购电补缺，物理可行不能等同于经济鲁棒，亦不证明停电情况下的供电韧性。

## 7. 可用于论文的表述

> 为检验所选方案的稳定性，在保持最终模型不变的前提下，开展参数邻域扰动、时间顺序重训回测、评分窗口调整及条件压力回放。融合与因果后处理参数变化下，净负荷RMSE相对基准变化为−0.241%至+0.698%；核心树参数微调后的变化为−0.972%至+0.661%。在2025年2—12月的各月份及30、60、90日滚动窗口中，问题二方案的费用均低于exp006基准，表明其改善并非仅集中于少数时间区间。另一方面，固定购电与模式计划对持续负荷上偏和光伏下偏具有明显成本敏感性，联合2%偏移使费用增加14.22%。因此，所选模型在所测参数和开发期时段上表现稳定，但持续系统性预测偏差仍是其经济表现的主要限制。由于该年度数据曾参与开发，以上结论属于开发期回测和条件性鲁棒性证据，不作为新年度独立泛化证明。

建议正文优先选图01、02、04、06、08、12；图03、05、07、09—11、13—15放附录或按篇幅选用。参数与时段稳定性、压力脆弱点均应保留，不能只展示有利结果。

## 8. 图像、数据和复现

{figure_table}

每组均提供 **450 dpi PNG、嵌入字体的矢量PDF、字形转路径的SVG**。LaTeX推荐PDF，Word可用PNG或SVG。图中中文、单位、时间范围与条件说明已保留；完整长图注在captions.md，图表数据在evidence下四个子目录。

脚本位于仓库 `experiments/exp008/robustness/`。从仓库根目录，在既有`.venv`环境依次执行：

```sh
.venv/bin/python -m experiments.exp008.robustness.forecast_analysis
.venv/bin/python -m experiments.exp008.robustness.forecast_hyperparameters
.venv/bin/python -m experiments.exp008.robustness.q1_analysis --resume
.venv/bin/python -m experiments.exp008.robustness.temporal_analysis
.venv/bin/python -m experiments.exp008.robustness.execution_analysis
.venv/bin/python -m experiments.exp008.robustness.build_figures
.venv/bin/python -m experiments.exp008.robustness.build_delivery
```

forecast脚本默认拒绝覆盖已存在的计算目录，重算时应先把本次 `data/results/exp008/robustness/forecast/` 移到另一个备份位置；不要移动或修改最终模型的原始冻结目录。仅重新导出图像时只需运行build_figures和build_delivery。读取说明、图片与CSV无需训练环境；真正重训需要本项目的原始CSV和已签名预测档案。60个拟合模型、完整预测数组与回放数组留在本地data目录，分享包保留图像、表格、协议和源码，不重复打包大型权重。
'''
    (OUT/'README.md').write_text(report)
    captions=['# 论文图注','']
    latex=['% 从 Xelatex/ 目录编译时的路径；从仓库根编译请去掉 ../。',
           '% 导言区需要 \\usepackage{graphicx}。按需要选用以下片段，不必全部插入。','']
    for row in figures:
        captions += [f"## 图{row['id'][:2]}：{row['title']}",'',row['caption'],'',
            '数据：'+ '、'.join(f'[对应表](evidence/{p})' for p in row['sources']),'']
        latex += ['\\begin{figure}[htbp]','  \\centering',
            f"  \\includegraphics[width=0.96\\linewidth]{{../paper/figures/exp008_robustness/{row['id']}.pdf}}",
            '  \\caption{'+row['title']+'}',f"  \\label{{fig:exp008-{row['id']}}}",'\\end{figure}','']
    (OUT/'captions.md').write_text('\n'.join(captions))
    (OUT/'figures.tex').write_text('\n'.join(latex))
    (FIG/'README.md').write_text('# exp008 最终成果的敏感性与鲁棒性图像\n\n'
        '15组图，每组PNG（450 dpi）、PDF、SVG；主模型与原结果保持冻结。\n\n'
        '[分析说明](../../../reports/experiments/exp008/robustness/README.md) · '
        '[全部图预览](overview.png) · [15页图集](all_figures.pdf) · '
        '[图注](../../../reports/experiments/exp008/robustness/captions.md)\n\n'
        '本目录专用于当前 exp008，不能与旧实验图像混用。\n')
    # Package evidence and image formats without duplicating the 60 local model weights.
    archive=OUT/'exp008_robustness_paper_assets.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for base in (FIG,OUT):
            for path in sorted(base.rglob('*')):
                if path.is_file() and path != archive and path.name != 'delivery_manifest.json':
                    z.write(path,path.relative_to(ROOT))
    manifest={str(p.relative_to(ROOT)):sha(p) for base in (FIG,OUT) for p in base.rglob('*')
              if p.is_file() and p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps({'files_sha256':manifest,'figure_groups':len(figures)},indent=2))
    print(json.dumps({'figure_groups':len(figures),'report':str(OUT/'README.md'),
                      'package':str(archive)},ensure_ascii=False))


if __name__=='__main__':
    main()
