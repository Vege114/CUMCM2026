"""Check numeric consistency across the durable report, figures and app snapshot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='exp001');args=p.parse_args()
    root=Path(__file__).resolve().parent
    report=root/'experiments'/args.experiment
    if json.loads((report/'protocol.json').read_text()).get('architecture', {}).get('name') == 'periodic_residual_mlp':
        from verify_delivery_v2 import verify
        verify(args.experiment)
        return
    snapshot=json.loads((report/'app/src/data.json').read_text())
    assert snapshot['buildStatus']=='complete' and len(snapshot['reportContent'])==8
    narrative_tables=0
    for section in snapshot['reportContent']:
        for block in section['blocks']:
            if block['type']=='table':
                narrative_tables+=1
                assert block['rows']==snapshot['queries'][block['queryId']]['rows']
                for row in block['rows']:
                    assert '| '+' | '.join(row.values())+' |' in section['markdown']
            else:
                assert not any(line.startswith('|') for line in block['markdown'].splitlines())
    body=(report/'report.md').read_text()
    assert '本节将由完成核验' not in body and '训练正在进行' not in body
    dispatch=pd.read_csv(report/'dispatch_metrics.csv',dtype={'scenario':str,'seed':str})
    daily=pd.read_csv(report/'daily_primary.csv',dtype={'scenario':str})
    summary=pd.DataFrame(snapshot['queries']['summary']['rows'])
    assert set(summary.scenario)=={'2','3','4-2','4-3'} and len(summary)==4
    metrics=('planned_kwh','final_kwh','emergency_kwh','planned_cost','up_cost','down_cost',
             'emergency_cost','total_cost','violations')
    for row in summary.to_dict('records'):
        scenario=row['scenario'];schedule='0+6+12+18' if scenario in ('3','4-3') else '0'
        source=dispatch[(dispatch.scenario==scenario)&(dispatch.variant=='selected')&
                        (~dispatch.known_price)&dispatch.corrected&(dispatch.update_schedule==schedule)]
        assert len(source)==1
        days=daily[daily.scenario==scenario]
        assert len(days)==334 and days.date.nunique()==334
        assert days.date.min()=='2025-02-01' and days.date.max()=='2025-12-31'
        for metric in metrics:
            np.testing.assert_allclose(row[metric],source.iloc[0][metric],rtol=1e-11,atol=1e-7)
            np.testing.assert_allclose(row[metric],days[metric].sum(),rtol=1e-11,atol=1e-7)
        assert f"{row['total_cost']:,.2f}" in body
    forecasts=pd.read_csv(report/'forecast_metrics.csv',dtype={'seed':str})
    annual=pd.read_csv(report/'annual_forecast_metrics.csv',dtype={'seed':str})
    for row in annual.to_dict('records'):
        source=forecasts[(forecasts.variant==row['variant'])&(forecasts.seed==row['seed'])&
                         (forecasts.target==row['target'])&(forecasts.population==row['population'])&
                         (forecasts.lead=='all')]
        assert set(source.month)==set(range(2,13))
        count=source.n.sum()
        assert count==row['n']
        if row['population']=='all':assert count==192168
        np.testing.assert_allclose(row['mae'],source.absolute_error_sum.sum()/count,rtol=1e-11)
        np.testing.assert_allclose(row['rmse'],np.sqrt(source.squared_error_sum.sum()/count),rtol=1e-11)
        np.testing.assert_allclose(row['wape_pct'],100*source.absolute_error_sum.sum()/source.actual_abs_sum.sum(),rtol=1e-11)
    app_annual=pd.DataFrame(snapshot['queries']['annual_forecast']['rows'])
    keys=['variant','seed','target','population']
    a=annual.sort_values(keys).reset_index(drop=True);b=app_annual.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys])
    np.testing.assert_allclose(a[['mae','rmse','wape_pct']],b[['mae','rmse','wape_pct']],atol=1e-10,rtol=1e-11)
    assert len(snapshot['queries']['cost_comparison']['rows'])==32
    specified=pd.read_csv(report/'specified_dates.csv',dtype={'scenario':str})
    assert len(specified)==16
    assert set(specified.date)=={'2025-03-20','2025-06-21','2025-09-23','2025-12-21'}
    np.testing.assert_allclose(specified.total_cost,pd.DataFrame(snapshot['queries']['specified_days']['rows']).total_cost,rtol=1e-11)
    verification=json.loads((report/'verification.json').read_text())
    for row in summary.to_dict('records'):
        np.testing.assert_allclose(row['total_cost'],verification[f"result{row['scenario']}.xlsx"]['total_cost'],rtol=1e-11)
    figures=['monthly-forecast-error','annual-cost-comparison','daily-emergency-energy','technical-route',
             'annual-three-error-metrics','seed-variation','cost-components','failure-case']
    assert all((report/'figures'/f'{name}.{extension}').stat().st_size>1000 for name in figures for extension in ('png','svg'))
    status={'status':'passed','primary_scenarios':4,'days_per_scenario':334,'intervals_per_day':144,
            'reviewed_narrative_tables':narrative_tables,
            'forecast_records_per_model_target':192168,'annual_forecast_groups':len(annual),
            'comparison_strategies':32,'specified_day_tables':16,
            'checks':['CSV ↔ app snapshot','daily ↔ annual sums','forecast error sums ↔ annual metrics',
                      'workbook verification ↔ report totals','Markdown displayed totals','figure files present'],
            'browser_checks':'Recorded separately after actual browser inspection.'}
    (report/'consistency_checks.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':main()
