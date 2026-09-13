"""Independent arithmetic and exported-point audit for frozen report evidence."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/experiments/exp008/evidence/battery'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    manifest=json.loads((OUT/'manifest.json').read_text())
    table=pd.read_csv(OUT/'battery_metrics_all.csv').set_index('series')
    audits={}
    for key,row in table.iterrows():
        df=pd.read_csv(OUT/key/'power_all_intervals.csv')
        p=df.net_battery_power_kw.to_numpy(); dp=np.diff(p)
        ends=pd.to_datetime(df.interval_end)
        assert len(df)==48096 and ends.is_unique and ends.is_monotonic_increasing
        assert ends.iloc[0]==pd.Timestamp('2025-02-01 00:10') and ends.iloc[-1]==pd.Timestamp('2026-01-01')
        assert (np.diff(ends.values)==np.timedelta64(10,'m')).all()
        assert df.delta_power_kw.isna().sum()==1 and np.isnan(df.delta_power_kw.iloc[0])
        np.testing.assert_allclose(df.delta_power_kw.iloc[1:],dp,atol=1e-8,rtol=0)
        np.testing.assert_allclose(p,df.charge_power_kw-df.discharge_power_kw,atol=1e-8,rtol=0)
        mode=np.where(abs(p)>6e-6,np.sign(p),0); active=mode[mode!=0]
        checks=dict(direction_reversals_nonidle=np.sum(active[1:]!=active[:-1]),
            episodes=np.sum((mode!=0)&(mode!=np.r_[0,mode[:-1]])),
            throughput_kwh=np.sum(df.charge_power_kw+df.discharge_power_kw)/6,
            mean_absolute_delta_kw=np.mean(abs(dp)),rms_delta_kw=np.sqrt(np.mean(dp*dp)),
            p95_absolute_delta_kw=np.quantile(abs(dp),.95),max_absolute_delta_kw=np.max(abs(dp)),
            total_variation_kw=np.sum(abs(dp)),
            direct_reversals=np.sum(((p[:-1]>1e-6)&(p[1:]<-1e-6))|((p[:-1]<-1e-6)&(p[1:]>1e-6))),
            power_limit_share=np.mean(abs(p)>=5000-1e-6),large_jump_share=np.mean(abs(dp)>1000+1e-6))
        for metric,value in checks.items():
            np.testing.assert_allclose(value,row[metric],atol=1e-7,rtol=1e-12,err_msg=f'{key}:{metric}')
        for date in manifest['random_dates']+manifest['specified_dates']:
            day=pd.read_csv(OUT/key/f'power_{date}.csv')
            offset=(pd.Timestamp(date)-pd.Timestamp('2025-02-01')).days*144
            pd.testing.assert_frame_equal(day,df.iloc[offset:offset+144].reset_index(drop=True),check_exact=False,atol=1e-8,rtol=0)
        audits[key]=dict(passed=True,rows=len(df),independent_metrics=len(checks),raw_csv_sha256=digest(OUT/key/'power_all_intervals.csv'))
    figure_checks=[]
    for item in manifest['plot_manifest']:
        path=ROOT/item['file']; assert digest(path)==item['sha256']
        if path.suffix=='.svg' and item['kind']!='historical_summary':
            tree=ET.parse(path)
            expected=item['points_per_series']
            paths=[len(re.findall(r'[ML] ',node.attrib.get('d',''))) for node in tree.iter('{http://www.w3.org/2000/svg}path')]
            count=paths.count(expected)
            minimum=1 if item['kind']=='deterministic' else 2 if item['kind']=='annual' else 8
            assert count>=minimum,(path,count,expected)
            figure_checks.append(dict(file=item['file'],curve_paths_with_all_points=count,points_per_curve=expected))
    comparisons=[]
    for new,old in [('q2_final','exp006_q2'),('q3_final','exp002_q3'),('q4_2_final','exp002_q4_2'),('q4_3_final','exp002_q4_3')]:
        for metric in [x for x in table.columns if x not in ('comparable_to_final_Q2','model_seed')]:
            before,after=float(table.loc[old,metric]),float(table.loc[new,metric])
            comparisons.append(dict(current_series=new,previous_series=old,metric=metric,previous=before,current=after,
                absolute_change=after-before,relative_change_pct=(100*(after-before)/abs(before)) if before else None))
    pd.DataFrame(comparisons).to_csv(OUT/'same_question_power_comparisons.csv',index=False)
    result=dict(passed=True,series=audits,SVG_point_audit=figure_checks,
        audit_source_sha256=digest(Path(__file__)),note='Independent CSV formulas and SVG path point counts; optimizer was not rerun.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=True,series=len(audits),SVG_point_checks=len(figure_checks))))

if __name__=='__main__': run()
