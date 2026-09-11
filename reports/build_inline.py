"""Fill the two reusable conversation visuals with verified experiment results."""

import argparse
import csv
import json
from pathlib import Path

NAMES={'selected':'按月验证选模','mlp':'多层感知机','gru':'门控循环网络','tcn':'因果卷积网络',
       'gru_no_calendar':'移除日历特征','gru_one_day':'一天历史窗口','yesterday':'昨日同期','weekly':'历史周同期'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='exp001')
    p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parent
    rows=[]
    with (root/'experiments'/args.experiment/'dispatch_metrics.csv').open() as f:
        for row in csv.DictReader(f):
            if row['seed'] not in ('mean','baseline') or row['known_price']!='False' or row['corrected']!='True':continue
            expected='0+6+12+18' if row['scenario'] in ('3','4-3') else '0'
            if row['update_schedule']!=expected:continue
            rows.append({'scenario':row['scenario'],'variant':row['variant'],'model_label':NAMES[row['variant']],
                         'total_cost':float(row['total_cost']),'emergency_kwh':float(row['emergency_kwh'])})
    assert len(rows)==32
    args.directory.mkdir(parents=True,exist_ok=True)
    for template,name in [('metrics.inline.html','microgrid-metrics.html'),('technical-path.inline.html','neural-path.html')]:
        fragment=(root/'templates'/template).read_text().replace('@@DATA@@',json.dumps(rows,ensure_ascii=False))
        assert '<html' not in fragment and '@@DATA@@' not in fragment
        destination=args.directory/name;destination.write_text(fragment)
        print(destination)


if __name__=='__main__':main()
