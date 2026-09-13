"""运行主方案及对照后：python q3_analysis.py，生成检验、题目指定表和论文图。
该脚本不复用旧版费用；所有结果由本次退款MILP输出逐项导出。
"""
import json  # 读取求解元数据。
import sys  # 保留异常的非零退出码。
from pathlib import Path  # 便携相对路径。
import numpy as np  # 独立数值核验与统计。
import pandas as pd  # 汇总与原始十分钟结果导出。
import matplotlib  # 无界面绘图，适用于命令行环境。
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # 论文折线、柱状、热力图。
from q3_reproduce import HERE, Forecasts, day_run, plan, fees, execute, verify, save, DT, TOL


def metric(actual,predicted):
    """统一计分口径，R²为相对实测均值的决定系数。"""
    error=np.asarray(predicted)-actual
    denominator=float(np.square(actual-np.mean(actual)).sum())
    return {'mae':float(np.abs(error).mean()),'mse':float(np.square(error).mean()),
            'rmse':float(np.sqrt(np.square(error).mean())),
            'r2':float(1-np.square(error).sum()/denominator) if denominator else None}


def structural_tests():
    """退款、未来扰动、冻结前缀与严格场景物理限制的针对性测试。"""
    np.testing.assert_allclose(fees(np.array([100.,100.]),np.array([80.,120.]),
                               np.zeros(2),np.ones(2)).sum(-1),[90,130])
    original=Forecasts(); changed=Forecasts(); checks=[]
    for day,slot in ((31,0),(78,36),(171,72),(264,108)):
        # 先比较当前预测与历史路径，再污染当前及未来实况、未发布官方预报。
        a=original.get(day,slot).copy(); pa,_=original.paths(day,slot)
        changed.actual=original.actual.copy()
        changed.actual.reshape(-1,2)[day*144+slot:]*=1.7
        changed.official={k:(v*1.4 if k>day*144+slot else v.copy()) for k,v in original.official.items()}
        changed.get.cache_clear(); changed.base.cache_clear()
        b=changed.get(day,slot); pb,_=changed.paths(day,slot)
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
        np.testing.assert_allclose(pa,pb,atol=1e-8,rtol=0)
        checks.append({'day':day,'slot':slot,'future_mutation_unchanged':True})
    actual=np.array([[1000.,0.],[0.,2000.],[1000.,0.]])
    q=np.array([100.,0.,100.]); before=execute(q,actual,6000.)
    modified=actual.copy(); modified[1:]*=2
    after=execute(q,modified,6000.)
    for key in ('charge','discharge','emergency','surplus','states'):
        np.testing.assert_allclose(before[key][0],after[key][0],atol=1e-10)
    prediction=np.column_stack((np.full(4,6000.),np.zeros(4)))
    _,meta=plan(prediction,np.array([.1,.1,1.5,1.5]),1200.,np.zeros((1,4)),
                final_day=True,seconds=5.)
    assert meta['scenario_cd_overlap']<TOL and meta['scenario_ce_overlap']<TOL
    return {'refund_example_passed':True,'causality_checks':checks,
            'execution_future_prefix_passed':True,'strict_scenario_physics_passed':True}


def load_archive(case):
    with np.load(HERE/'results'/case/'dispatch_all.npz',allow_pickle=False) as archive:
        data={key:archive[key].copy() for key in archive.files}
    verify(data)  # 图表之前先核验全轨迹。
    if len(data['final'])!=365: raise ValueError('论文图只接受完整365日回放')
    return data


def main():
    output=HERE/'paper'; output.mkdir(exist_ok=True)
    a=load_archive('main'); b=load_archive('no_update'); f=Forecasts()
    checks=structural_tests(); rows=[]
    # 四季指定日采用主方案各自真实日初状态，作同状态局部参数压力测试。
    dates=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    days=[(pd.Timestamp(date)-pd.Timestamp('2025-01-01')).days for date in dates]
    sensitivity=[]
    for scale in (.9,1.,1.1):
        total=0.; emergency=0.; maxima=[]
        for day in days:
            detail,logs=day_run(f,day,float(a['states'][day,0]),scale=scale)
            verify({k:v[None,...] for k,v in detail.items()})
            total+=float(detail['fees'].sum()); emergency+=float(detail['emergency'].sum())
            maxima.append(max(x['gap'] for x in logs))
        sensitivity.append({'scale':scale,'four_day_cost':total,'emergency_kwh':emergency,'max_gap':max(maxima)})
    center=sensitivity[1]['four_day_cost']
    for row in sensitivity: row['cost_change_pct']=100*(row['four_day_cost']/center-1)
    # 不重叠地对每次发布后的6小时计分，避免提前量长的日子重复加权。
    truth=[]; updated=[]; midnight=[]
    for day in range(31,365):
        first=f.get(day,0)
        for slot in (0,36,72,108):
            truth.append(f.actual[day,slot:slot+36])
            updated.append(f.get(day,slot)[:36])
            midnight.append(first[slot:slot+36])
    truth,updated,midnight=[np.concatenate(x) for x in (truth,updated,midnight)]
    scores={}
    for name,p in [('updated',updated),('midnight',midnight)]:
        scores[name]={key:metric(y,pred) for key,y,pred in
                      [('load',truth[:,0],p[:,0]),('pv',truth[:,1],p[:,1]),
                       ('net',truth[:,0]-truth[:,1],p[:,0]-p[:,1])]}
    for day in range(31,365):
        date=str(pd.Timestamp('2025-01-01')+pd.Timedelta(days=day))[:10]
        for t in range(144):
            rows.append({'date':date,'slot':t+1,'start_hour':t/6,
                **{k:float(a[k][day,t]) for k in ('original','final','charge','discharge','emergency','surplus','price')},
                'soc_start':float(a['states'][day,t]),'soc_end':float(a['states'][day,t+1]),
                'cost_yuan':float(a['fees'][day,t].sum())})
    pd.DataFrame(rows).to_csv(output/'all_intervals.csv',index=False,encoding='utf-8-sig')
    specified=[]; storage=[]; emergencies=[]
    for date,day in zip(dates,days):
        row={'date':date,'original_total':float(a['original'][day].sum()),
             'final_total':float(a['final'][day].sum()),'cost_yuan':float(a['fees'][day].sum()),
             'soc_start':float(a['states'][day,0]),'soc_end':float(a['states'][day,-1])}
        for hour in (10,12,14,16,18,20):
            row[f'original_{hour}']=float(a['original'][day,hour*6])
            row[f'final_{hour}']=float(a['final'][day,hour*6])
        specified.append(row)
        for block in range(6):
            storage.append({'date':date,'start_hour':4*block,'end_hour':4*(block+1),
                'charge_kwh':float(a['charge'][day,block*24:(block+1)*24].sum()),
                'discharge_kwh':float(a['discharge'][day,block*24:(block+1)*24].sum())})
        mask=a['emergency'][day]>TOL; t=0
        while t<144:
            if not mask[t]: t+=1; continue
            start=t
            while t<144 and mask[t]: t+=1
            emergencies.append({'date':date,'start':f'{start//6:02d}:{start%6*10:02d}',
                'end':f'{t//6:02d}:{t%6*10:02d}','emergency_kwh':float(a['emergency'][day,start:t].sum())})
    for name,data in [('specified',specified),('storage',storage),('emergency',emergencies),('sensitivity',sensitivity)]:
        pd.DataFrame(data).to_csv(output/f'{name}.csv',index=False,encoding='utf-8-sig')
    summary=json.loads((HERE/'results/main/summary.json').read_text(encoding='utf-8'))
    baseline=json.loads((HERE/'results/no_update/summary.json').read_text(encoding='utf-8'))
    comparison={'main_cost':summary['total_cost'],'midnight_cost':baseline['total_cost'],
        'saving_yuan':baseline['total_cost']-summary['total_cost'],
        'saving_pct':100*(1-summary['total_cost']/baseline['total_cost']),
        'formal_initial_soc_main':summary['initial_soc'],'formal_initial_soc_baseline':baseline['initial_soc'],
        'scope':'Each policy starts Jan1 at6000 and warms independently; Feb initial inventory differs, disclosed.'}
    save(output/'analysis.json',{'tests':checks,'forecast_metrics':scores,'sensitivity':sensitivity,
         'comparison':comparison,'specified':specified,'storage':storage,'emergencies':emergencies})
    # 统一字体、单位和印刷分辨率；图注由TeX正文给出。
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'savefig.dpi':300})
    months=pd.date_range('2025-02-01','2025-12-31').month.to_numpy()
    fig,ax=plt.subplots(figsize=(8.3,3.8),layout='constrained')
    for data,label,color in [(a,'日内更新','#136B80'),(b,'仅午夜计划','#A95735')]:
        cost=data['fees'][31:].sum((1,2))
        ax.plot(range(2,13),[cost[months==m].sum()/1e4 for m in range(2,13)],'o-',label=label,color=color)
    ax.set(title='不同信息更新策略的月度实际费用',xlabel='月份',ylabel='实际费用 / 万元',xticks=range(2,13))
    ax.grid(alpha=.2); ax.legend(); fig.savefig(output/'q3_monthly.png'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained',sharex=True)
    for ax,date,day in zip(axes.ravel(),dates,days):
        x=np.arange(144)/6
        ax.plot(x,a['original'][day],color='#A95735',label='原计划')
        ax.plot(x,a['final'][day],color='#136B80',label='最终购电')
        for h in (6,12,18): ax.axvline(h,color='.65',ls=':',lw=.8)
        ax.set(title=date,xlabel='时刻 / h',ylabel='区间购电量 / kWh',xticks=[0,6,12,18,24]); ax.grid(alpha=.15)
    axes[0,0].legend(); fig.suptitle('四个指定日的购电计划调整'); fig.savefig(output/'q3_purchase.png'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained',sharex=True)
    for ax,date,day in zip(axes.ravel(),dates,days):
        ax.plot(np.arange(144)/6,6*(a['charge'][day]-a['discharge'][day]),color='#136B80',label='净充电功率')
        ax.axhline(0,color='.5',lw=.7)
        twin=ax.twinx(); twin.plot(np.arange(145)/6,a['states'][day],color='#B76B30',alpha=.8,label='储电量')
        twin.set(ylim=(0,12000),ylabel='储电量 / kWh'); twin.axhline(1200,color='#B76B30',ls=':',lw=.6)
        twin.axhline(10800,color='#B76B30',ls=':',lw=.6)
        ax.set(title=date,xlabel='时刻 / h',ylabel='电池功率 / kW',ylim=(-5500,5500),xticks=[0,6,12,18,24])
    axes[0,0].legend(loc='upper left',fontsize=8); twin.legend(loc='upper right',fontsize=8)
    fig.suptitle('指定日电池功率与储电量'); fig.savefig(output/'q3_storage.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.3,3.8),layout='constrained')
    heat=a['final'][31:]-a['original'][31:]; scale=np.quantile(np.abs(heat),.99)
    image=ax.imshow(heat,aspect='auto',cmap='RdBu_r',vmin=-scale,vmax=scale,extent=(0,24,334,0))
    ax.set(title='年度最终购电相对原计划的调整',xlabel='时刻 / h',ylabel='评价期日序 / 日（2月1日起）',xticks=[0,6,12,18,24])
    fig.colorbar(image,ax=ax,label='净调整量 / kWh'); fig.savefig(output/'q3_adjustment.png'); plt.close(fig)
    print(json.dumps({'comparison':comparison,'sensitivity':sensitivity,'forecast_metrics':scores},ensure_ascii=False))


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,AssertionError) as exc:
        print(f'分析停止：{exc}',file=sys.stderr); sys.exit(1)
