"""Derive battery power from frozen execution archives; no solver or training.

Run: python -m reports.battery_power_exp004
"""
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/experiments/exp004'
SEED = 20260912
LABELS = {'exp003': 'exp003 正式基线', 'no_season': 'exp004 无季节',
          'causal_season': 'exp004 历史季节（正式）',
          'oracle_season': 'exp004 全年探索（含未来）'}
COLORS = ['#69747b', '#357c89', '#aa6d37', '#8b7c9e']
BEGIN, END = '<!-- battery-power:start -->', '<!-- battery-power:end -->'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(z, days):
    c, d, s = z['charge'], z['discharge'], z['states']
    assert c.shape == d.shape == (days, 144) and s.shape == (days, 145)
    assert all(np.isfinite(a).all() for a in (c, d, s))
    assert min(c.min(), d.min()) >= -1e-7
    assert max(c.max(), d.max()) * 6 <= 5000 + 1e-5
    assert not ((c > 1e-6) & (d > 1e-6)).any()
    assert s.min() >= 1200-1e-6 and s.max() <= 10800+1e-6
    np.testing.assert_allclose(np.diff(s, axis=1), np.sqrt(.9)*c-d/np.sqrt(.9), atol=1e-6, rtol=0)
    np.testing.assert_allclose(s[:-1, -1], s[1:, 0], atol=1e-6, rtol=0)


def metrics(p):
    p = np.asarray(p).ravel()
    delta = np.diff(p)
    a = abs(delta)
    return {'points': len(p), 'transitions': len(delta),
            'mean_abs_delta_kw': float(a.mean()), 'rms_delta_kw': float(np.sqrt(np.mean(delta**2))),
            'p95_abs_delta_kw': float(np.quantile(a, .95)), 'max_abs_delta_kw': float(a.max()),
            'total_variation_kw': float(a.sum()), 'jump_gt_1000_pct': float(100*np.mean(a > 1000)),
            'direct_reversals': int(np.sum(((p[:-1]>1e-6)&(p[1:] < -1e-6)) | ((p[:-1]<-1e-6)&(p[1:]>1e-6)))),
            'at_power_limit_pct': float(100*np.mean(abs(p) >= 5000-1e-6))}


def figure(powers, dates, annual=False):
    """SVG retains one vertex per input point, without simplification or smoothing."""
    width, height, panel = 1100, 1160, 260
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           '<title>电池充放电功率：充电为正，放电为负</title>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Microsoft YaHei,Arial,sans-serif" font-size="15" fill="#333">',
           f'<text x="80" y="27">电池充放电功率 · {"2025 全年所有 52,560 时点/方案" if annual else "随机四日 · 抽样种子 20260912"}</text>']
    panels = list(powers) if annual else dates
    for i, item in enumerate(panels):
        top, bottom, left, right = 75+i*panel, 250+i*panel, 85, 1070
        y = lambda v: bottom-(v+5500)/11000*(bottom-top)
        title = LABELS[item] if annual else str(item.date())
        out.append(f'<text x="{left}" y="{top-15}">{title}</text>')
        if annual:
            edge = left+(right-left)*31/365
            out.append(f'<rect x="{left}" y="{top}" width="{edge-left}" height="{bottom-top}" fill="#eee"/>')
            out.append(f'<text x="{edge+8}" y="{top+18}" font-size="12">← 1月共用预热｜2–12月评价 →</text>')
        for tick in [-5000, -2500, 0, 2500, 5000]:
            out.extend([f'<line x1="{left}" x2="{right}" y1="{y(tick)}" y2="{y(tick)}" stroke="{ "#888" if tick==0 else "#ddd"}"/>',
                        f'<text x="{left-8}" y="{y(tick)+5}" text-anchor="end">{tick}</text>'])
        out.append(f'<text transform="translate(20,{(top+bottom)/2}) rotate(-90)" text-anchor="middle">净功率 / kW</text>')
        ticks = [(0,'1月'),(59/365,'3月'),(120/365,'5月'),(181/365,'7月'),(243/365,'9月'),(304/365,'11月'),(1,'年末')] if annual else [(h/24,f'{h:02}:00') for h in [0,4,8,12,16,20,24]]
        for fraction,label in ticks:
            out.append(f'<text x="{left+(right-left)*fraction}" y="{bottom+23}" text-anchor="middle">{label}</text>')
        out.append(f'<text x="{(left+right)/2}" y="{bottom+44}" text-anchor="middle">{"2025年区间终点（10分钟）" if annual else "区间终点时刻（24:00为次日00:00）"}</text>')
        keys = [item] if annual else list(powers)
        for key in keys:
            values = powers[key].ravel() if annual else powers[key][item.dayofyear-1]
            points = ' '.join(f'{left+(right-left)*(j+1)/len(values):.4f},{y(v):.4f}' for j,v in enumerate(values))
            idx = list(LABELS).index(key)
            dash = ' stroke-dasharray="5 3"' if key == 'exp003' else ''
            out.append(f'<polyline data-series="{key}" data-points="{len(values)}" points="{points}" fill="none" stroke="{COLORS[idx]}" stroke-width="{.55 if annual else 1.4}"{dash}/>')
    for i, (key,label) in enumerate(LABELS.items()):
        x = 65+(i%2)*510; yy = 1120+(i//2)*25
        out.extend([f'<line x1="{x}" x2="{x+30}" y1="{yy}" y2="{yy}" stroke="{COLORS[i]}" stroke-width="3"/>', f'<text x="{x+38}" y="{yy+5}">{label}</text>'])
    return ''.join(out)+ '</g></svg>'


def replace_section(text, section):
    return re.sub(re.escape(BEGIN)+'.*?'+re.escape(END), '', text, flags=re.S).rstrip()+ '\n\n'+BEGIN+'\n'+section+'\n'+END+'\n'


def build(attach=True):
    evidence, figures = REPORT/'evidence', REPORT/'figures'
    evidence.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    sources = {}
    warm_path = ROOT/'data/results/exp003/warmup_2.npz'
    with np.load(warm_path) as z:
        warm = {k:z[k] for k in z.files}
    validate(warm,31); sources[str(warm_path.relative_to(ROOT))] = sha(warm_path)
    powers, rows, tables = {}, [], []
    times = pd.date_range('2025-01-01 00:10', periods=52560, freq='10min')
    dates = sorted(pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(i)) for i in np.random.default_rng(SEED).choice(np.arange(31,365),4,replace=False))
    joins = {}
    for key in LABELS:
        path = ROOT/('data/results/exp003/dispatch_2.npz' if key=='exp003' else f'data/results/exp004/dispatch_{key}_seed_42.npz')
        with np.load(path) as z:
            validate(z,334)
            np.testing.assert_allclose(warm['states'][-1,-1], z['states'][0,0], atol=1e-6, rtol=0)
            c = np.concatenate([warm['charge'],z['charge']])*6
            d = np.concatenate([warm['discharge'],z['discharge']])*6
        sources[str(path.relative_to(ROOT))]=sha(path)
        p = c-d; powers[key]=p
        for scope, values in [('evaluation',p[31:]),('full_year_with_warmup',p)]:
            rows.append({'variant':key,'label':LABELS[key],'scope':scope,**metrics(values)})
        joins[key] = float(p[31,0]-p[30,-1])
        flat = p.ravel()
        tables.append(pd.DataFrame({'variant':key,'model_seed':42,'interval_end':times,
             'date':np.repeat(pd.date_range('2025-01-01',periods=365).strftime('%Y-%m-%d'),144),
             'phase':np.repeat(['shared_warmup']*31+['evaluation']*334,144),
             'charge_kw':c.ravel(),'discharge_kw':d.ravel(),'net_charge_kw':flat,
             'delta_power_kw':np.r_[np.nan,np.diff(flat)]}))
    full = pd.concat(tables,ignore_index=True)
    full.to_csv(evidence/'battery_power_full_year.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    full[full.date.isin([str(d.date()) for d in dates])].to_csv(evidence/'battery_power_random_days.csv',index=False)
    stats = pd.DataFrame(rows); stats.to_csv(evidence/'battery_power_metrics.csv',index=False)
    ev = stats[stats.scope=='evaluation'].set_index('variant')
    compare=[]
    for key in list(LABELS)[1:]:
        for metric in metrics(powers[key][31:]):
            if metric in ('points','transitions'): continue
            before,after=float(ev.loc['exp003',metric]),float(ev.loc[key,metric])
            compare.append({'variant':key,'metric':metric,'baseline':before,'current':after,'absolute_change':after-before,'relative_change_pct':100*(after-before)/abs(before) if before else None})
    pd.DataFrame(compare).to_csv(evidence/'battery_power_comparison.csv',index=False)
    for annual,name in [(False,'battery-power-random-days'),(True,'battery-power-full-year')]:
        (figures/f'{name}.svg').write_text(figure(powers,dates,annual),encoding='utf-8')
    cols={'label':'方案','mean_abs_delta_kw':'平均|ΔP| / kW','rms_delta_kw':'ΔP均方根 / kW','p95_abs_delta_kw':'|ΔP| P95 / kW','max_abs_delta_kw':'最大|ΔP| / kW','jump_gt_1000_pct':'大于1000 kW / %','direct_reversals':'直接反转次数','at_power_limit_pct':'贴限 / %'}
    table = ev.reset_index()[list(cols)].rename(columns=cols).to_html(index=False,float_format=lambda x:f'{x:,.2f}',border=0)
    formal=ev.loc['causal_season']; base=ev.loc['exp003']
    change=100*(formal.mean_abs_delta_kw/base.mean_abs_delta_kw-1)
    conclusion=(f'未观察到功率变化被明显抑制。正式历史季节组的平均相邻功率变化为 {formal.mean_abs_delta_kw:.2f} kW，较 exp003 变化 {change:+.2f}%；'
                f'P95 为 {formal.p95_abs_delta_kw:.2f} kW，最大跳变 {formal.max_abs_delta_kw:.2f} kW。'
                '现有执行器仅按供需余额、SOC 和 5000 kW 上限充放电，没有显式爬坡约束或平滑惩罚。'
                '波动是否下降应结合下表各项判断；功率贴限不代表变化率受限，此比较也不能证明平滑机制的因果效果或统计显著性。')
    note=('P=6×(充电量−放电量)，母线侧 kW；充电为正、放电为负。每点对应前10分钟平均功率，使用区间终点。'
          '1月为共用预热，2–12月为各方案独立执行的正式评价，模型种子42。全年每方案52,560点，评价48,096点；'
          '评价差分48,095项，含跨午夜变化，不含预热到评价的跳变（另存清单）。'
          '大跳变阈值1000 kW/10min仅作描述，零容差1e-6 kW；未做降采样、移动平均或曲线平滑。'
          '随机日使用NumPy default_rng(20260912)，从334个评价日不放回抽4天：'+ '、'.join(str(d.date()) for d in dates)+'。')
    body=f'<h1>电池充放电功率与波动核验</h1><p>{note}</p><p>{conclusion}</p><h2>2–12月同口径波动指标</h2><div class="table">{table}</div>'
    for name,title in [('battery-power-random-days','随机四天'),('battery-power-full-year','全年所有时点')]:
        body+=f'<h2>{title}</h2>'+ (figures/f'{name}.svg').read_text(encoding='utf-8')
    body+='<p>数据：<a href="evidence/battery_power_full_year.csv.gz">全年原始CSV（gzip）</a> · <a href="evidence/battery_power_random_days.csv">抽样日CSV</a> · <a href="evidence/battery_power_metrics.csv">全年及评价指标</a> · <a href="evidence/battery_power_comparison.csv">基线差值</a> · <a href="evidence/battery_power_manifest.json">来源与核验</a></p>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>exp004 电池功率核验</title><style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;color:#333;max-width:1180px;margin:32px auto;padding:0 20px}svg{width:100%;height:auto}table{border-collapse:collapse;min-width:950px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}.table{overflow-x:auto}h1,h2{font-weight:500}</style><body>'+body+'</body></html>'
    (REPORT/'battery-power.html').write_text(page,encoding='utf-8')
    inline_data={'series':[{'key':k,'label':LABELS[k]} for k in ['exp003','causal_season']],
                 'days':[{'date':str(day.date()),'values':{k:np.round(powers[k][day.dayofyear-1],3).tolist() for k in ['exp003','causal_season']}} for day in dates]}
    template=(ROOT/'reports/templates/battery-power.inline.html').read_text(encoding='utf-8')
    (REPORT/'inline-battery-power.html').write_text(template.replace('__BATTERY_EVIDENCE__',json.dumps(inline_data,ensure_ascii=False)),encoding='utf-8')
    manifest={'status':'passed','sampling_seed':SEED,'sampling_algorithm':'numpy.random.default_rng.choice without replacement, then date sort','numpy_version':np.__version__,'model_seed':42,'sample_dates':[str(d.date()) for d in dates],'points_per_variant':52560,'evaluation_points_per_variant':48096,'variants':LABELS,'sources':sources,'warmup_to_evaluation_delta_kw':joins,'checks':['finite values','shape and complete regular time grid','power bounds','charge/discharge exclusivity','SOC equation including efficiencies','SOC limits','cross-day continuity','warmup/evaluation SOC continuity'],'conclusion':conclusion,'reproduction':'python -m reports.battery_power_exp004'}
    manifest['outputs']={str(p.relative_to(REPORT)):sha(p) for p in [REPORT/'battery-power.html',*figures.glob('battery-power-*.svg'),*evidence.glob('battery_power*.csv*')]}
    (evidence/'battery_power_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if attach:
        path=REPORT/'report.html'
        content=re.sub(re.escape(BEGIN)+'.*?'+re.escape(END),'',path.read_text(encoding='utf-8'),flags=re.S)
        # Self-contained appendix: copied HTML continues to work without the sidecar.
        appendix=BEGIN+'<section style="max-width:1180px;margin:24px auto;padding:20px;font-family:sans-serif"><h2>补充核验：电池充放电功率</h2><p>'+html.escape(conclusion)+'</p><details><summary>展开随机日、全年原始曲线及波动指标</summary><iframe title="电池充放电功率核验" style="width:100%;height:1200px;border:0" srcdoc="'+html.escape(page,quote=True)+'"></iframe></details><p><a href="battery-power.html">单独打开完整电池功率图</a></p></section>'+END
        content=re.sub(r'(<body[^>]*>)',lambda m:m[0]+appendix,content,count=1)
        path.write_text(content,encoding='utf-8')
        md=REPORT/'report.md'
        md.write_text(replace_section(md.read_text(encoding='utf-8'),'## 补充：电池充放电功率与波动核验\n\n'+note+'\n\n'+conclusion+'\n\n[完整指标与曲线](battery-power.html)\n\n![随机四日](figures/battery-power-random-days.svg)\n\n![全年所有时点](figures/battery-power-full-year.svg)'),encoding='utf-8')
        metadata=REPORT/'report_build.json'
        record=json.loads(metadata.read_text(encoding='utf-8'))
        record['html_sha256']=sha(path)
        record['battery_power_addendum']={'manifest':'evidence/battery_power_manifest.json','sha256':sha(evidence/'battery_power_manifest.json')}
        metadata.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dates':manifest['sample_dates'],'evaluation':ev.reset_index().to_dict('records')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    build()
