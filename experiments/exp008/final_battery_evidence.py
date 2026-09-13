"""Reproducible final battery evidence and static figures, with no dispatch changes."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.exp008.power_diagnostics import prepare
from experiments.exp008.verify import battery_metrics
from experiments.exp008.frozen_sources import exp005_dispatch, final_template, TEMPLATE_MAIN_COMMIT

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'reports/experiments/exp008/evidence/battery'
FIGURES = ROOT / 'reports/experiments/exp008/figures/battery'
SEED = 20260912
DATES = pd.date_range('2025-02-01', '2025-12-31')
RANDOM = np.sort(np.random.default_rng(SEED).choice(334, 4, replace=False))
OFFICIAL_DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
OFFICIAL = np.array([DATES.get_loc(pd.Timestamp(day)) for day in OFFICIAL_DATES])
BLUE, GRAY, GOLD = '#245a82', '#7d8187', '#a56b27'
SOURCES = {
    'q2_final': ('data/results/exp008/mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days/dispatch.npz', '2', '第二问最终方案', 'data/results/exp003/warmup_2.npz'),
    'q3_final': ('data/results/exp008/absolute_hgb_update_dispatch/full334/3/dispatch_3.npz', '3', '第三问最终方案', 'data/results/exp002/warmup_3.npz'),
    'q4_2_final': ('data/results/exp008/q4_hgb_linked_price_physical/full334/dispatch_4-2.npz', '4-2', '第四问·日前方案', 'data/results/exp002/warmup_4-2.npz'),
    'q4_3_final': ('data/results/exp008/absolute_hgb_update_dispatch/full334/4-3/dispatch_4-3.npz', '4-3', '第四问·更新方案', 'data/results/exp002/warmup_4-3.npz'),
    'exp002_q2': ('data/results/exp002/dispatch_2.npz', '2', 'exp002 正式策略', 'data/results/exp002/warmup_2.npz'),
    'exp003_q2': ('data/results/exp003/dispatch_primary_seed_42.npz', '2', 'exp003 正式策略', 'data/results/exp003/warmup_2.npz'),
    'exp004_q2': ('data/results/exp004/dispatch_causal_season_seed_42.npz', '2', 'exp004 正式因果季节策略', 'data/results/exp003/warmup_2.npz'),
    'exp006_q2': ('data/results/exp006/primary/dispatch_2.npz', '2', 'exp006 正式策略', 'data/results/exp003/warmup_2.npz'),
    'exp002_q3': ('data/results/exp002/dispatch_3.npz', '3', 'exp002 第三问', 'data/results/exp002/warmup_3.npz'),
    'exp002_q4_2': ('data/results/exp002/dispatch_4-2.npz', '4-2', 'exp002 第四问·日前', 'data/results/exp002/warmup_4-2.npz'),
    'exp002_q4_3': ('data/results/exp002/dispatch_4-3.npz', '4-3', 'exp002 第四问·更新', 'data/results/exp002/warmup_4-3.npz'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load(path):
    with np.load(path) as z:
        return {key: z[key].copy() for key in z.files}


def physical_checks(a, eta=np.sqrt(.9), allow_overlap=False, allow_emergency_charging=False):
    c, d, s = a['charge'], a['discharge'], a['states']
    assert c.shape == d.shape == (334, 144) and s.shape == (334, 145)
    assert all(np.isfinite(value).all() for value in a.values() if isinstance(value, np.ndarray))
    assert min(c.min(), d.min()) >= -1e-6 and max(c.max(), d.max()) <= 5000/6 + 1e-6
    assert s.min() >= 1200 - 1e-6 and s.max() <= 10800 + 1e-6
    np.testing.assert_allclose(s[1:, 0], s[:-1, -1], atol=1e-6, rtol=0.)
    state_error = float(np.abs(np.diff(s, axis=1) - eta*c + d/eta).max())
    assert state_error < 1e-6
    overlap = int(np.sum((c > 1e-6) & (d > 1e-6)))
    emergency_charging = int(np.sum((c > 1e-6) & (a['emergency'] > 1e-6)))
    if not allow_overlap:
        assert overlap == 0
    if not allow_emergency_charging:
        assert emergency_charging == 0
    return dict(physical_tolerance_kwh=1e-6, maximum_SOC_equation_error_kwh=state_error,
        overlap_intervals=overlap, emergency_charging_intervals=emergency_charging,
        eta_charge=eta, eta_discharge=eta, all334days_continuous=True)


def metric_row(key, a, metric, *, comparable=True):
    b = battery_metrics(a)
    return dict(series=key, comparable_to_final_Q2=comparable, model_seed=42,
        cost_yuan=float(a['fees'].sum()), charge_starts=b['charge_starts'], discharge_starts=b['discharge_starts'],
        episodes=b['charge_starts']+b['discharge_starts'], throughput_kwh=b['throughput_kwh'],
        equivalent_full_cycles=b['equivalent_full_cycles'], active_slots=b['active_slots'],
        direction_reversals_nonidle=b['direction_reversals'], initial_soc=b['initial_soc'], final_soc=b['final_soc'],
        **{k: metric[k] for k in ('mean_absolute_delta_kw','rms_delta_kw','p95_absolute_delta_kw',
            'max_absolute_delta_kw','total_variation_kw','direct_reversals','power_limit_share',
            'large_jump_share','warmup_boundary_delta_kw')})


def raw_export(a, directory):
    """Descriptive historical export for a different physical protocol."""
    directory.mkdir(parents=True, exist_ok=True)
    end = pd.date_range('2025-02-01 00:10', periods=48096, freq='10min')
    power = 6*(a['charge']-a['discharge']).ravel()
    delta = np.diff(power)
    frame = pd.DataFrame(dict(interval_start=end-pd.Timedelta(minutes=10), interval_end=end,
        charge_power_kw=6*a['charge'].ravel(), discharge_power_kw=6*a['discharge'].ravel(),
        net_battery_power_kw=power, delta_power_kw=np.r_[np.nan,delta],
        soc_start_kwh=a['states'][:,:-1].ravel(),soc_end_kwh=a['states'][:,1:].ravel(),
        load_kw=a['actual'][...,0].ravel(),pv_kw=a['actual'][...,1].ravel()))
    frame.to_csv(directory/'power_all_intervals.csv',index=False)
    for day in np.unique(np.r_[RANDOM,OFFICIAL]):
        frame.iloc[day*144:(day+1)*144].to_csv(directory/f'power_{DATES[day]:%Y-%m-%d}.csv',index=False)
    m = dict(mean_absolute_delta_kw=float(np.abs(delta).mean()),rms_delta_kw=float(np.sqrt(np.mean(delta**2))),
        p95_absolute_delta_kw=float(np.quantile(np.abs(delta),.95)),max_absolute_delta_kw=float(np.abs(delta).max()),
        total_variation_kw=float(np.abs(delta).sum()),direct_reversals=int(np.sum((power[:-1]>1e-6)&(power[1:]<-1e-6)
            | (power[:-1]<-1e-6)&(power[1:]>1e-6))),power_limit_share=float(np.mean(np.abs(power)>=5000-1e-6)),
        large_jump_share=float(np.mean(np.abs(delta)>1000+1e-6)),warmup_boundary_delta_kw=None)
    return frame,m


def style():
    path=Path('/System/Library/Fonts/STHeiti Light.ttc')
    fm.fontManager.addfont(str(path))
    matplotlib.rcParams.update({'font.family':fm.FontProperties(fname=str(path)).get_name(),
        'axes.unicode_minus':False,'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#51545a',
        'axes.labelcolor':'#30343b','text.color':'#30343b','xtick.color':'#51545a','ytick.color':'#51545a',
        'path.simplify':False,'agg.path.chunksize':0,'svg.fonttype':'none','savefig.dpi':180})


def power_axis(ax):
    ax.axhline(0,color='#52565d',lw=.7)
    for value in (-5000,5000): ax.axhline(value,color='#a8abb0',ls=':',lw=.7)
    ax.set_ylim(-5450,5450);ax.set_yticks([-5000,0,5000]);ax.set_ylabel('净电池功率 / kW')
    ax.grid(axis='y',color='#e5e7eb',lw=.5);ax.spines[['top','right']].set_visible(False)


def figure_save(fig,name,ledger,series,kind,points):
    for extension in ('png','svg'):
        path=FIGURES/f'{name}.{extension}'
        fig.savefig(path,bbox_inches='tight',facecolor='white')
        ledger.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),kind=kind,series=series,
            points_per_series=points,annual_downsampling=False,smoothing=False))
    plt.close(fig)


def daily_plot(frames,new,old,indices,tag,ledger):
    fig,axes=plt.subplots(2,2,figsize=(14,8.4),sharex=True,sharey=True)
    for ax,day in zip(axes.flat,indices):
        for key,color,line,label in ((old,GRAY,'--',SOURCES[old][2]),(new,BLUE,'-',SOURCES[new][2])):
            values=frames[key].iloc[day*144:(day+1)*144].net_battery_power_kw.to_numpy()
            artist=ax.plot(np.arange(1,145)/6,values,color=color,ls=line,lw=1.05,label=label)[0]
            assert len(artist.get_xdata())==144
        power_axis(ax);ax.set_title(f'{DATES[day]:%Y-%m-%d}',loc='left')
        ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24],['00:00','06:00','12:00','18:00','24:00'])
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False)
    subtitle=f'固定随机种子 {SEED}；不放回抽取并排序' if tag=='random' else '题目指定日期；与随机日另外展示'
    fig.suptitle(f'{SOURCES[new][2]}：完整十分钟实测执行功率',x=.07,ha='left',fontsize=15)
    fig.text(.07,.02,f'{subtitle}。充电为正、放电为负；00:10 表示 00:00–00:10 的区间终点。每条每日曲线保留全部 144 点。',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.10,hspace=.27,wspace=.15)
    figure_save(fig,f'{new}_{tag}_days',ledger,[old,new],tag,144)


def annual_plot(frames,new,old,ledger):
    fig,axes=plt.subplots(2,1,figsize=(15,7.5),sharex=True,sharey=True)
    for ax,key,color in zip(axes,(old,new),(GRAY,BLUE)):
        df=frames[key]
        line=ax.plot(pd.to_datetime(df.interval_end),df.net_battery_power_kw,color=color,lw=.25)[0]
        assert len(line.get_xdata())==48096
        power_axis(ax);ax.set_title(SOURCES[key][2],loc='left')
    ticks=pd.to_datetime(['2025-02-01','2025-04-01','2025-06-01','2025-08-01','2025-10-01','2026-01-01'])
    axes[-1].set_xticks(ticks,['02-01','04-01','06-01','08-01','10-01','12-31 24:00'])
    axes[-1].set_xlim(pd.Timestamp('2025-02-01'),pd.Timestamp('2026-01-01'))
    fig.suptitle(f'{SOURCES[new][2]}：334 日正式评价全部 48,096 点',x=.065,ha='left',fontsize=15)
    fig.text(.065,.025,'2025-02-01 至 2025-12-31；预热不拼入评分。无抽稀、无平滑；SVG 与原始 CSV 可查看全部十分钟时点。',fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.12,hspace=.29)
    figure_save(fig,f'{new}_all_48096_points',ledger,[old,new],'annual',48096)


def run():
    EVIDENCE.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    template_ref=TEMPLATE_MAIN_COMMIT
    template_hashes={}
    for name in ('battery-power.md','history-comparison.md'):
        content, template_provenance=final_template('reports/templates/'+name,root=ROOT)
        (EVIDENCE/name).write_bytes(content);template_hashes[name]=hashlib.sha256(content).hexdigest()
    sources,frames,rows={}, {}, []
    for key,(relative,scenario,label,warmup_relative) in SOURCES.items():
        path,warmup_path=ROOT/relative,ROOT/warmup_relative
        a,warmup=load(path),load(warmup_path)
        initial_soc=float(warmup['states'][-1,-1]);initial_power=float(6*(warmup['charge'][-1,-1]-warmup['discharge'][-1,-1]))
        np.testing.assert_allclose(initial_soc,a['states'][0,0],atol=1e-8,rtol=0.)
        own=physical_checks(a)
        directory=EVIDENCE/key
        if not (directory/'manifest.json').exists():
            metric=prepare(path,directory,scenario=scenario,initial_soc=initial_soc,
                initial_power_kw=initial_power,seed=SEED,role='user accepted final' if key.endswith('final') else 'historical registered primary')
        else:
            metric=json.loads((directory/'power_metrics.json').read_text())
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['source_sha256']==sha(path)
        assert manifest['random_dates']==[DATES[i].strftime('%Y-%m-%d') for i in RANDOM]
        manifest.update(warmup_boundary_source_verified=True,warmup_source=str(warmup_path.relative_to(ROOT)),
            warmup_source_sha256=sha(warmup_path),warmup_last_SOC_matches_formal_first_SOC=True,
            warmup_last_power_kw=initial_power,warmup_boundary_reference='verified preceding warmup archive final AC power',
            independent_power_checks=own,model_seed=42,specified_dates=OFFICIAL_DATES)
        save(directory/'manifest.json',manifest)
        df=pd.read_csv(directory/'power_all_intervals.csv')
        assert len(df)==48096 and df.delta_power_kw.isna().sum()==1 and pd.isna(df.delta_power_kw.iloc[0])
        np.testing.assert_allclose(df.net_battery_power_kw,6*(a['charge']-a['discharge']).ravel(),atol=1e-8,rtol=0.)
        for day in OFFICIAL:
            df.iloc[day*144:(day+1)*144].to_csv(directory/f'power_{DATES[day]:%Y-%m-%d}.csv',index=False)
        frames[key]=df;rows.append(metric_row(key,a,metric,comparable=scenario=='2'))
        sources[key]=manifest
    # Original exp001 is preserved but explicitly not on the same efficiency/initial-state protocol.
    a=load(ROOT/'data/results/exp001/dispatch_2.npz')
    own=physical_checks(a,eta=.9)
    df,metric=raw_export(a,EVIDENCE/'exp001_original_q2');frames['exp001_original_q2']=df
    r=metric_row('exp001_original_q2',a,metric,comparable=False)
    r['model_seed']='mean(42,2026,3407)'
    r['equivalent_full_cycles']=(.9*a['charge'].sum()+a['discharge'].sum()/.9)/(2*12000)
    rows.append(r)
    sources['exp001_original_q2']=dict(source='data/results/exp001/dispatch_2.npz',source_sha256=sha(ROOT/'data/results/exp001/dispatch_2.npz'),
        model_seed='mean(42,2026,3407)',comparable=False,reason='Original one-way efficiency .9 and different warmup SOC; descriptive only',physical_checks=own)
    # Read exp005 directly from its immutable original branch, not from exp007 results.
    content, historical_source=exp005_dispatch(ROOT)
    path=EVIDENCE/'exp005_primary_source.npz';path.write_bytes(content)
    a=load(path);assert a['charge'].shape==(334,144)
    own=physical_checks(a,allow_overlap=True,allow_emergency_charging=True)
    df,metric=raw_export(a,EVIDENCE/'exp005_q2');frames['exp005_q2']=df
    rows.append(metric_row('exp005_q2',a,metric,comparable=False))
    sources['exp005_q2']=dict(**historical_source,source_sha256=sha(path),comparable=False,
        reason='Additional 1000kW/10min hard ramp and emergency charging allowed; beta=.1 original result, descriptive only',physical_checks=own)
    # Q1 powers are already kW; never multiply these components by six again.
    q1path=ROOT/'data/results/exp008/q1/revised.json'
    result=json.loads(q1path.read_text());q={k:np.asarray(v) for k,v in result['trajectory'].items()}
    q1power=q['c']-q['d'];assert len(q1power)==144
    np.testing.assert_allclose(np.diff(q['E']),np.sqrt(.9)*q['c']/6-q['d']/(6*np.sqrt(.9)),atol=1e-8,rtol=0.)
    q1frame=pd.DataFrame(dict(interval_start_hour=np.arange(144)/6,interval_end_hour=np.arange(1,145)/6,
        charge_power_kw=q['c'],discharge_power_kw=q['d'],net_battery_power_kw=q1power,
        delta_power_kw=np.r_[np.nan,np.diff(q1power)],soc_start_kwh=q['E'][:-1],soc_end_kwh=q['E'][1:]))
    q1frame.to_csv(EVIDENCE/'q1_deterministic_144_points.csv',index=False)
    sources['q1']=dict(source=str(q1path.relative_to(ROOT)),source_sha256=sha(q1path),points=144,
        input_power_units='already kW',role='Appendix1 deterministic planned schedule, not annual measured execution',
        original_metrics=result['metrics'])
    metrics=pd.DataFrame(rows);metrics.to_csv(EVIDENCE/'battery_metrics_all.csv',index=False)
    previous=metrics.set_index('series').loc['exp006_q2'];current=metrics.set_index('series').loc['q2_final']
    comparisons=[]
    for metric in ('cost_yuan','direction_reversals_nonidle','episodes','active_slots','throughput_kwh','equivalent_full_cycles',
        'mean_absolute_delta_kw','rms_delta_kw','p95_absolute_delta_kw','max_absolute_delta_kw','total_variation_kw',
        'direct_reversals','power_limit_share','large_jump_share'):
        old,new=float(previous[metric]),float(current[metric])
        comparisons.append(dict(metric=metric,previous=old,current=new,absolute_change=new-old,
            relative_change_pct=100*(new-old)/abs(old) if old else None,comparable=True))
    pd.DataFrame(comparisons).to_csv(EVIDENCE/'q2_vs_exp006_power_comparison.csv',index=False)
    style();ledger=[]
    pairs=[('q2_final','exp006_q2'),('q3_final','exp002_q3'),('q4_2_final','exp002_q4_2'),('q4_3_final','exp002_q4_3')]
    for new,old in pairs:
        daily_plot(frames,new,old,RANDOM,'random',ledger)
        daily_plot(frames,new,old,OFFICIAL,'specified',ledger)
        annual_plot(frames,new,old,ledger)
    fig,ax=plt.subplots(figsize=(14,4.5));line=ax.plot(q1frame.interval_end_hour,q1power,color=BLUE,lw=1.3)[0]
    assert len(line.get_xdata())==144
    power_axis(ax);ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24],['00:00','06:00','12:00','18:00','24:00'])
    ax.set_title('第一问：附录 1 确定性调度功率（完整 144 点）',loc='left',fontsize=14)
    fig.text(.08,.015,'这是一日计划，不是全年实测曲线。原数据充、放电分量已经为 kW；充电为正、放电为负。',fontsize=9)
    fig.subplots_adjust(bottom=.17);figure_save(fig,'q1_deterministic_power',ledger,['q1'],'deterministic',144)
    # Historical battery measures, ordered by experiment; no ranking/lines for incompatible protocols.
    labels=['exp001 原登记*','exp002','exp003','exp004 因果季节','exp005 β=0.1*','exp006','exp008 最终']
    keys=['exp001_original_q2','exp002_q2','exp003_q2','exp004_q2','exp005_q2','exp006_q2','q2_final']
    selected=metrics.set_index('series').loc[keys]
    fig,axes=plt.subplots(1,3,figsize=(16,6.6),sharey=True)
    for ax,(metric,title,scale) in zip(axes,[('direction_reversals_nonidle','非空方向反转 / 次',1),
        ('episodes','连续充/放活动段 / 次',1),('throughput_kwh','总吞吐 / 百万 kWh',1e6)]):
        for i,(key,row) in enumerate(selected.iterrows()):
            incompatible=key in ('exp001_original_q2','exp005_q2')
            value=float(row[metric])/scale
            ax.barh(i,value,color='white' if incompatible else BLUE if key=='q2_final' else '#b7c7d5',
                edgecolor=GRAY if incompatible else BLUE,linewidth=.7,hatch='///' if incompatible else None,height=.56)
            ax.text(value,i,f'  {value:,.0f}' if scale==1 else f'  {value:.3f}',va='center',fontsize=9)
        ax.set_title(title,loc='left');ax.set_xlim(0,float(selected[metric].max()/scale)*1.20)
        ax.grid(axis='x',color='#e5e7eb',lw=.5);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.set_yticks(np.arange(len(labels)),labels)
    axes[0].invert_yaxis()
    fig.suptitle('第二问电池行为：保留历史正式策略及不可比原登记',x=.08,ha='left',fontsize=15)
    fig.text(.08,.035,'334 日、48,096 时段。* exp001 效率/初始状态不同；exp005 额外爬坡约束并允许紧急电充电；仅描述，不计算跨口径改善率。',fontsize=9)
    fig.subplots_adjust(left=.145,top=.89,bottom=.12,wspace=.22)
    figure_save(fig,'q2_history_battery_intensity',ledger,keys,'historical_summary',48096)
    # Numerical plotting checks and exact CSV binding; first formal difference stays missing.
    all_sources={key:row.get('source_sha256') for key,row in sources.items()}
    manifest=dict(template_main_revision=template_ref,template_sha256=template_hashes,
        source_manifest=sources,source_hashes=all_sources,formal_days=334,formal_points=48096,
        complete365_trace_claimed=False,random_seed=SEED,random_offsets=RANDOM.tolist(),
        random_dates=[DATES[i].strftime('%Y-%m-%d') for i in RANDOM],specified_dates=OFFICIAL_DATES,
        algorithm='numpy.random.default_rng(20260912).choice(334,4,replace=False); sort',
        plot_manifest=ledger,script_sha256=sha(Path(__file__)),reproduce='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp008.final_battery_evidence',
        numerical_checks_passed=True,visual_QA_status='pending rendered inspection',
        user_accepted_final_before_original8percent_target=True,
        notes=['No scenario uses another question as a same-scenario fee comparator.',
            'Q1 is one deterministic schedule, distinct from annual measured execution.',
            'Annual PNG/SVG paths contain all48096 points; path simplification is disabled.',
            'All listed experiments are preserved; exp007 is excluded.'])
    save(EVIDENCE/'manifest.json',manifest)
    print(json.dumps(dict(random_dates=manifest['random_dates'],figures=len(ledger),evidence=str(EVIDENCE),
        final_Q2_cost=float(current.cost_yuan),final_Q2_reversals=int(current.direction_reversals_nonidle)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
