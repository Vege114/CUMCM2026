"""将已验证的同一份计算结果转为LaTeX表格及工作簿中间载荷。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from q3_reproduce import HERE, save, TOL


def table(caption,label,headers,rows,fmt=None):
    fmt=fmt or ('l'+'r'*(len(headers)-1))
    lines=[r'\begin{table}[htbp]',r'\centering\small',
           '\\caption{'+caption+'}\\label{'+label+'}',
           '\\begin{tabular}{'+fmt+'}',r'\toprule',
           ' & '.join(headers)+r'\\',r'\midrule']
    lines.extend(' & '.join(str(x) for x in row)+r'\\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    return '\n'.join(lines)+'\n'


def figure(name,title,label,note,width='.96'):
    return ('\\begin{figure}[htbp]\n\\centering\n'
            f'\\includegraphics[width={width}\\textwidth]{{q3_run/paper/{name}.png}}\n'
            f'\\caption{{{title}}}\\label{{{label}}}\n'
            f'\\parbox{{.94\\textwidth}}{{\\footnotesize 图注：{note}}}\n\\end{{figure}}\n')


def main():
    out=HERE/'paper'
    a=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    s=json.loads((HERE/'results/main/summary.json').read_text(encoding='utf-8'))
    b=json.loads((HERE/'results/no_update/summary.json').read_text(encoding='utf-8'))
    with np.load(HERE/'results/main/dispatch_all.npz') as z: data={k:z[k][31:].copy() for k in z.files}
    f=lambda x:f'{x:,.2f}'
    result=(f'本次退款口径下的实际总费用为{f(s["total_cost"])}元，紧急购电量为'
            f'{f(s["emergency_kwh"])}\\,kWh，下调取消电量为{f(s["downward_kwh"])}\\,kWh。'
            '表\\ref{tab:q3-cost}将退款与支出分别列出，全部费用由实际执行轨迹逐槽结算。\n')
    result+=table('第三问334日实际费用分解','tab:q3-cost',['项目','金额/元'],
        [[label,f(value)] for label,value in zip(['原计划购电费','上调追加费','下调退款（扣减）','紧急购电费'],s['components'])]
        +[['总费用',f(s['total_cost'])]])
    result+=table('四个指定日的计划购电与实际总费用','tab:q3-daily',
        ['日期','原计划/kWh','最终常规购电/kWh','实际总费用/元'],
        [[r['date'][5:],f(r['original_total']),f(r['final_total']),f(r['cost_yuan'])] for r in a['specified']])
    result+=table('指定十分钟区间的原计划与最终购电量（kWh）','tab:q3-slots',
        ['日期','时段','原计划','最终购电'],
        [[r['date'][5:],f'{h}:00--{h}:10',f(r[f'original_{h}']),f(r[f'final_{h}'])]
         for r in a['specified'] for h in (10,12,14,16,18,20)],'llrr')
    result+=table('指定日期的四小时充放电量（kWh）','tab:q3-storage',
        ['日期','时段','充电量','放电量'],
        [[r['date'][5:],f'{r["start_hour"]}:00--{r["end_hour"]}:00',f(r['charge_kwh']),f(r['discharge_kwh'])]
         for r in a['storage']],'llrr')
    result+=table('指定日期的日初与日末储电量','tab:q3-soc',
        ['日期','0:00储电量/kWh','24:00储电量/kWh'],
        [[r['date'][5:],f(r['soc_start']),f(r['soc_end'])] for r in a['specified']])
    # 所有连续区间完整列出；使用同类浮动表保证编号按出现顺序排列。
    result+=table('指定日期的完整紧急购电区间','tab:q3-emergency',
        ['日期','时间段','紧急购电量/kWh'],
        [[r['date'][5:],r['start']+'--'+r['end'],f(r['emergency_kwh'])]
         for r in a['emergencies']],'llr')
    result+=('四小时汇总内充、放电量同时为正，表示不同十分钟动作的累积，不代表同槽充放电。'
             '下调退款使部分过量采购可以撤回，但退款仅为原款的一半，提前购电仍有误差损失。'
             '紧急购电由当槽供需、实际SOC和额定功率共同决定，不能只按单槽预测误差大小解释。\n')
    result+=figure('q3_purchase','指定日原计划与最终购电曲线','fig:q3-purchase',
       '同一轨迹展示四季指定日的计划调整，竖线标出预报发布时间，曲线差额对应最终净调整。')
    result+=figure('q3_storage','指定日电池净功率与储电量','fig:q3-storage',
       '正功率为充电、负功率为放电，双轴保留全部十分钟动作，储电量始终受安全边界约束。')
    result+=figure('q3_adjustment','年度购电调整时序分布','fig:q3-heat',
       '暖色表示上调、冷色表示退款下调，色限取绝对调整量的百分之九十九分位，极值仅作显示截断。')
    result+=('完整334日结果保存为\\texttt{q3\\_run/result3.xlsx}，十分钟原始轨迹见'
             '\\texttt{q3\\_run/paper/all\\_intervals.csv}。\n')
    (out/'q3_results.tex').write_text(result,encoding='utf-8')
    validation=table('午夜保持与六小时更新的预测误差','tab:q3-prediction',
        ['变量','版本','MAE/kW','MSE/$\\mathrm{kW}^2$','RMSE/kW','$R^2$'],
        [[label,version,f(a['forecast_metrics'][key][variable]['mae']),
          f(a['forecast_metrics'][key][variable]['mse']),f(a['forecast_metrics'][key][variable]['rmse']),
          f"{a['forecast_metrics'][key][variable]['r2']:.4f}"]
         for variable,label in [('load','负载'),('pv','光伏'),('net','净需求')]
         for key,version in [('midnight','午夜保持'),('updated','日内更新')]],'llrrrr')
    check=s['verification']
    validation+=table('规划场景与实际执行的独立核验','tab:q3-verification',['检查项目','结果'],
        [['最大供需残差/kWh',f"{check['balance_kwh']:.2e}"],
         ['最大SOC递推残差/kWh',f"{check['state_kwh']:.2e}"],
         ['最大逐槽计费残差/元',f"{check['billing_yuan']:.2e}"],
         ['跨日SOC断点/kWh',f"{check['cross_day_kwh']:.2e}"],
         ['实际同时充放电槽数',str(check['simultaneous_slots'])],
         ['实际紧急电充电槽数',str(check['emergency_charging_slots'])],
         ['所有规划场景最大充放重叠/kWh',f"{s['max_scenario_cd_overlap']:.2e}"],
         ['所有规划场景最大充电与紧急电重叠/kWh',f"{s['max_scenario_ce_overlap']:.2e}"],
         ['求解次数（含一月）',str(s['solves'])],
         ['平均/最大相对间隙',f"{100*s['mean_gap']:.3f}\\% / {100*s['max_gap']:.3f}\\%"],
         ['限时结束次数',str(s['time_limit_solves'])]],'lr')
    validation+=('全部实际轨迹通过物理与计费核验，但部分规划限时终止，不能据此认定经济最优。'
                 '日内目标省略已付原计划常数，并包含退款和终端库存项；其相对间隙可能较大，'
                 '不等同于年度真实费用的相对误差。'
                 f'仅午夜对照的平均与最大间隙分别为{100*b["mean_gap"]:.3f}\\%和'
                 f'{100*b["max_gap"]:.3f}\\%；主方案与对照的完整回放用时分别为'
                 f'{s["wall_seconds"]:.2f}\\,s和{b["wall_seconds"]:.2f}\\,s，均不含HGB训练。\n')
    validation+=table('误差路径尺度的四季指定日敏感性','tab:q3-sensitivity',
        ['$\\kappa$','四日总费用/元','相对变化/\\%','紧急购电量/kWh'],
        [[f"{r['scale']:.1f}",f(r['four_day_cost']),f"{r['cost_change_pct']:+.3f}",f(r['emergency_kwh'])]
         for r in a['sensitivity']])
    maximum=max(abs(r['cost_change_pct']) for r in a['sensitivity'])
    validation+=(f'在上述四个指定日及同日初状态条件下，误差尺度上下浮动10\\%时，费用最大偏移为'
                 f'{maximum:.3f}\\%。这一结果衡量局部风险参数敏感性；固定秒数预算下可行解也可能变化，'
                 '不将有限日期的结果推广为任意天气或跨年度的稳定性保证。\n')
    validation+=table('相同预测输入层下的信息更新策略对照','tab:q3-comparison',
        ['方案','实际总费用/元','紧急购电/kWh','非空换向/次'],
        [['仅午夜计划',f(b['total_cost']),f(b['emergency_kwh']),str(b['reversals'])],
         ['六小时滚动更新',f(s['total_cost']),f(s['emergency_kwh']),str(s['reversals'])]])
    comp=a['comparison']
    validation+=(f'完整策略对照中，日内更新相对仅午夜计划减少费用{f(comp["saving_yuan"])}元，'
                 f'变化比例为{comp["saving_pct"]:.3f}\\%。两组均从1月1日6000\\,kWh开始独立预热，'
                 f'2月1日初始库存分别为{s["initial_soc"]:.3f}与{b["initial_soc"]:.3f}\\,kWh，'
                 '因此该比较反映包括预热和后续重规划在内的完整策略差异，不是固定二月初态的单因素因果估计。'
                 '是否保留每一次发布，还需在相同起点下逐次删除更新机会；本实验不据总体比较断言每个更新时间都必要。\n')
    validation+=figure('q3_monthly','日内更新与仅午夜计划的月度实际费用','fig:q3-monthly',
        '按相同结算公式逐槽累加月度费用，两组独立传递实际库存，展示更新策略的季节性费用差异。',width='.86')
    validation+=('两组单次规划均限时2秒，但更新方案每天求解4次，午夜对照仅求解1次；'
                 '且限时MILP的最优间隙针对各自代理目标，不能作为年度实际费用的最优性保证。'
                 '因此上述差额是给定求解预算下的策略实测差异，并不把信息、计算投入和解质量的影响完全分离。'
                 '2025年数据已用于此前模型开发，故本次属于按时间推进的开发年度评价。'
                 '物理核验和未来扰动检查支持实现的一致性与信息边界，但不构成未触碰测试集、全局最优或电池延寿证明。\n')
    (out/'q3_validation.tex').write_text(validation,encoding='utf-8')
    # 工作簿保持附件结构，所有数值来自同一a；费用核验列由导出器写成求和公式。
    dates=pd.date_range('2025-02-01','2025-12-31')
    hours=lambda t:f'{t//6:02d}:{t%6*10:02d}'
    headers=['日期\\时间']+[f'{hours(t)}--{hours(t+1)}' for t in range(144)]+['全天购电量/kWh','全天购电费/元']
    payload={'headers':headers,'dates':[str(x.date()) for x in dates],
             'original':data['original'].tolist(),'final':data['final'].tolist(),
             'components':data['fees'].sum(1).tolist(),'price':data['price'][0].tolist(),
             'storage':[],'emergency':[]}
    for i,date in enumerate(payload['dates']):
        for j in range(6):
            payload['storage'].append([date if j==0 else None,f'{4*j}:00--{4*j+4}:00',
                float(data['charge'][i,j*24:(j+1)*24].sum()),float(data['discharge'][i,j*24:(j+1)*24].sum()),
                '0:00' if j==0 else ('24:00' if j==1 else None),
                float(data['states'][i,0]) if j==0 else (float(data['states'][i,-1]) if j==1 else None)])
        t=0; first=True
        while t<144:
            if data['emergency'][i,t]<=TOL: t+=1; continue
            start=t
            while t<144 and data['emergency'][i,t]>TOL: t+=1
            payload['emergency'].append([date if first else None,f'{hours(start)}--{hours(t)}',
                                         float(data['emergency'][i,start:t].sum())]); first=False
    save(out/'workbook_payload.json',payload)


if __name__=='__main__': main()
