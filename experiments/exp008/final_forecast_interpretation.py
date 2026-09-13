"""Supplemental report prose and local visual-QA record from stable forecast CSVs."""
import json
import xml.etree.ElementTree as ET

import pandas as pd

from experiments.exp008.final_forecast_evidence import OUT, FIG, ROOT, sha, save_json


def run():
    annual = pd.read_csv(OUT/'annual_metrics.csv')
    monthly = pd.read_csv(OUT/'monthly_metrics.csv')
    daily = pd.read_csv(OUT/'daily_metrics.csv')
    energy = pd.read_csv(OUT/'daily_energy_metrics.csv')
    leads = pd.read_csv(OUT/'lead_metrics.csv')
    stages = pd.read_csv(OUT/'stage_metrics.csv')
    configuration = json.loads((OUT/'model_configuration.json').read_text())
    training = pd.read_csv(OUT/'training_models_and_boundaries.csv')
    deltas, rises = [], []
    for target in ('load','pv','net_load'):
        rows = annual[(annual.target==target)&(annual.population=='all')].set_index('model')
        for key in ('mae_kw','rmse_kw','wape_pct'):
            value, baseline = rows.loc['final_blend',key],rows.loc['single_hgb',key]
            deltas.append({'target':target,'metric':key,'blend':value,'single_HGB':baseline,
                'delta_blend_minus_HGB':value-baseline,'relative_delta_pct':100*(value/baseline-1)})
        rows = monthly[(monthly.target==target)&(monthly.population=='all')].pivot(index='month',columns='model',values='rmse_kw')
        for month, row in rows.iterrows():
            if row.final_blend>row.single_hgb:
                rises.append({'target':target,'month':int(month),'blend_rmse_kw':row.final_blend,
                    'single_hgb_rmse_kw':row.single_hgb,'increase_kw':row.final_blend-row.single_hgb})
    worst = daily[(daily.model=='final_blend')&(daily.target=='net_load')&(daily.population=='all')].nlargest(5,'rmse_kw')
    e = energy[(energy.model=='final_blend')&(energy.target=='net_load')].copy()
    e['abs_energy_error_kwh']=e.signed_energy_error_kwh.abs()
    extreme=leads[(leads.model=='final_blend')&(leads.target=='pv')&(leads.population=='all')].nlargest(5,'wape_pct')
    interpretation = {'source_csv_sha256':{f:sha(OUT/f) for f in ('annual_metrics.csv','monthly_metrics.csv','daily_metrics.csv','daily_energy_metrics.csv','lead_metrics.csv','stage_metrics.csv')},
        'annual_deltas_vs_single_HGB':deltas,'monthly_RMSE_increases_preserved':rises,
        'worst5_net_daily_RMSE':worst[['date','mae_kw','rmse_kw','wape_pct']].to_dict('records'),
        'worst5_absolute_net_daily_energy_error':e.nlargest(5,'abs_energy_error_kwh')[['date','signed_energy_error_kwh']].to_dict('records'),
        'largest5_PV_lead_WAPE_from_small_actual_denominators':extreme[['lead_slot','lead_end_inclusive_minutes','wape_pct','actual_absolute_sum_kw','absolute_error_sum_kw']].to_dict('records'),
        'no_missing_or_extreme_values_replaced':True,'no_new_model_or_dispatch':True}
    save_json(OUT/'interpretation.json',interpretation)
    current=annual[(annual.model=='final_blend')&(annual.population=='all')]
    names={'load':'负载','pv':'光伏','net_load':'净负荷'}
    def table(frame, cols):
        header='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
        for _,row in frame.iterrows():
            header+='| '+' | '.join(f'{row[c]:.6f}' if isinstance(row[c],float) else str(row[c]) for c in cols)+' |\n'
        return header
    notes='''# 预测方法与证据：报告可引用正文

## 模型对应范围

第二问最终调度使用固定等权的 HGB–ExtraTrees 原始预测融合，随后重新执行 Ridge28 残差校准及半强度日负载记忆。第三、四问已有完整核验的运行采用单 HGB 加同样后处理。两类预测在本目录分开评分；这里的曲线与指标只对应午夜发布的未来24小时，不是第三、四问全部发布时间的合并评分。第一问按题面给定的单日时序求解，不把年度预测器写入第一问。

## 因果特征、训练边界及架构

每个十分钟槽使用31个手工构造特征，包含前一日、前一周的负载/光伏/净负荷，过去3日及7日的同槽统计量，时刻及星期周期项，以及前一日的日均值、标准差、最小值和最大值。每个特征只读取该次发布之前的实际值。最终模型没有CNN结构，没有把全年季节形状或未发布的未来实际值作为输入。

对2月至12月的每个月，分别为负载和光伏拟合一个 HGB 与一个 ExtraTrees，共44个原始回归器。设当月首日为以1月1日为0的第D日，训练集为第7日至D−8日，验证集为D−7日至D−1日；模型在该月冻结，后续每日特征随已观测历史更新。训练样本年龄权重半衰期为90日。HGB使用先前7日显式验证、10轮耐心与1e−7容差进行早停，保留最终拟合迭代。ExtraTrees保持同一验证边界，但验证只用于记录指标，不用于早停或再拟合。

HGB的固定参数为平方损失、最多15个叶节点、最多100轮、叶节点最少50个样本、L2正则2、学习率0.08。ExtraTrees为128棵树、叶节点最少10个样本、max_features=1、bootstrap=False、平方损失、n_jobs=2。所有原始模型固定seed42；仅有一个正式种子，不报告虚构的多种子标准差或置信区间。44个模型文件、训练/验证边界与hash列在training_models_and_boundaries.csv中。

两个raw预测在进入后处理前已被截为非负；光伏的允许时段由此前28个完整日实际光伏为正的槽位并集确定，并向两侧各扩展20分钟。因此这里的raw是“校准之前的因果原始输出”，不是完全没有物理后处理的树输出。

## raw融合与残差校准顺序

对每一日d、时段t、负载/光伏通道c，先计算

`raw_blend[d,t,c] = 0.5 raw_HGB[d,t,c] + 0.5 raw_ET[d,t,c]`。

权重在这个候选运行前已固定，两通道都用同一算术平均，没有逐日选择、权重网格搜索或对校准后的两个预测做混合。对融合raw自己的已发布历史重新拟合 Ridge28。它为负载、光伏分别拟合残差回归，但每个回归都包含另一通道的raw预测特征。最多使用此前28个已发布日，残差权重半衰期14日；10维输入由截距、raw、自身前一日/前一周/过去3日与raw的差、另一通道raw、四个时刻周期项组成。功率相关项以1000缩放，截距惩罚为2，其余系数为20；光伏raw或实际值大于1kW的历史样本权重为1，其余为0.1，再乘年龄权重。校准修正限制在±max(100kW,3倍历史残差RMSE)内，随后施加非负与同一因果光伏时段约束。

最终负载预测为

`load_final[d,t] = max(0, load_ridge[d,t] + 0.5 mean_t(actual_load[d−1,t]−load_ridge[d−1,t]))`。

光伏保持Ridge28输出。此处上一日残差始终相对于“尚未施加记忆的上一日Ridge28输出”，所以记忆不递归。2月1日没有更早的已发布预测，校准及记忆按冷启动规则保持原输出；用于调度历史情景的1月冷启动另有周期预测标识。

## 评价口径与结果

评价期为2025年2月1日至12月31日，共334个午夜发布、48,096个十分钟预测槽。误差定义为预测减实际；MAE为绝对误差总和除以样本数，RMSE为平方误差均值开方，WAPE为100倍绝对误差总和除以实际值绝对值总和。年度指标由逐槽误差聚合，不平均各月RMSE或WAPE。发电时段以实际PV>0作事后评分筛选，共26,880个槽，负载/光伏/净负荷都使用同一筛选；该当日实际筛选不参与预测。

第二问最终融合的全天指标如下（kW、%）：

'''
    notes+=table(current,['target','n','mae_kw','rmse_kw','wape_pct'])
    notes+='\n各月并非全部改善。对比单HGB，负载6月RMSE上升1.400641kW；光伏2、3、8、9、11月分别上升0.250634、1.291668、4.875315、2.434514、0.266381kW；净负荷3月上升0.112060kW。其余月度值及全部144个提前槽保留在CSV中。\n\n'
    notes+='净负荷RMSE最大的5日如下；这类错误日不能用年度均值掩盖：\n\n'
    notes+=table(worst,['date','mae_kw','rmse_kw','wape_pct'])
    notes+='\n6月1日的净负荷日能量预测误差为−18,827.088356kWh，即预测偏低。它是开发期失败案例，不对原因作未经验证的气象或节假日归因。\n\n'
    notes+='光伏少数提前槽的实际功率和接近零，使WAPE异常放大。例如第124槽（20:40）累计实际绝对值仅0.0149kW、累计绝对误差2349.560677kW，WAPE约15,768,863.6%。这不是缺失或计算错误，CSV保留原数值，提前量图对该面板明确使用对数轴。实际分母为0时WAPE留空。应连同RMSE、MAE及发电时段指标一起解释，而不能单看这些近零分母槽的百分比。\n\n'
    notes+='''## 历史实验、耗时与证据边界

exp001/002的历史原登记为多次发布时间合并口径，不能直接与本次午夜指标比较。history_original_registry.csv保留原登记160条记录；history_midnight_comparable.csv另列既有午夜重评分与正式午夜评分。exp004、005、006复用同一无季节预测；exp005来自固定git提交8108d0475ae4f2ea25565a806dfec5e836f62c0a中的真实原登记，只有4项预测值，发电时段负载/净负荷原登记缺失，留空而不按复用关系推算。exp007按用户指令排除。预测分数相同不意味着控制器、物理参数或费用口径也相同。

原始training_seconds字段的计时边界为model.fit之前到随后验证集model.predict之后，包含两个调用及周边开销。22个主模型对应计时之和：HGB为44.971846877秒，ExtraTrees为63.233388833秒；它们不是纯训练耗时。原pre-bridge工作流墙钟时间分别为52.487478958秒和83.763140041秒，包含各自原运行中的相关审计，也不是单独推理时间。纯fit、正式预测、Ridge28、记忆等阶段没有分开的原始timer，均保留NULL，不用本次报告脚本运行时间冒充，也不重新跑模型制造计时。

verification.json复算两个模型全部年度指标，核对月/日/144槽的可加分子汇总，逐元素核对raw的固定0.5融合与所选Q2调度实际引用的预测，检查44个原始模型文件hash，并引用已有独立因果审计。本次只读报告统计没有再次训练模型，也没有重新运行优化。独立审计是实现与数值复核，不等于独立留出测试：2025已被多次开发实验观察，不能声称这些指标是未触碰测试集的泛化性能。

最终Q2费用为13,201,981.947942元，较exp006下降6.1443%，去空闲后的方向换向数为2533次。用户已接受这个结果，因此报告如实记录6.1443%，不宣称达到原8%目标；也不把第二问的融合收益外推到仍用单HGB的第三、四问。
'''
    (OUT/'report_notes.md').write_text(notes)
    svg_checks=[]
    for path in sorted(FIG.glob('*.svg')):
        tree=ET.parse(path)
        namespace={'svg':'http://www.w3.org/2000/svg'}
        assert len(tree.findall('.//svg:path',namespace))>0
        svg_checks.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),
            'XML_parses':True,'embedded_glyph_paths':True,'render_inspection':'Not separately rasterized; matching PNG inspected.'})
    save_json(OUT/'visual_verification.json',{'passed_for_reviewed_PNG_exports':True,
        'PNG_files':[{'file':str(x.relative_to(ROOT)),'sha256':sha(x)} for x in sorted(FIG.glob('*.png'))],
        'PNG_review':'All5 final PNGs visually inspected through view_image. Chinese text readable, labels unclipped, axes/units visible, RMSE/WAPE panels and daily series correspond to named CSVs.',
        'repair':'Initial nonexistent PingFang path produced missing glyphs; replaced with existing Arial Unicode MS and re-rendered all figures. PV lead WAPE log axis explicitly labelled to retain near-zero-denominator spikes.',
        'encoding':'Annual bars use hatch; monthly lines use marker/line styles; lead lines use dash; history categories are explicitly labelled. Blue/gold labels distinguish models without color alone.',
        'scope_limits':'No color-vision simulation or independent SVG rasterization performed. Static figures reviewed at full export layouts, not a responsive dashboard.',
        'SVG_structure_checks':svg_checks})
    manifest=json.loads((OUT/'output_manifest.json').read_text())
    manifest['files_sha256']={x.name:sha(x) for x in OUT.iterdir() if x.is_file() and x.name!='output_manifest.json'}
    save_json(OUT/'output_manifest.json',manifest)
    print('Wrote interpretation.json, report_notes.md and visual_verification.json; no core6 file changed.')


if __name__=='__main__':
    run()
