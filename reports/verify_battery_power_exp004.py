"""Read exported evidence back independently and check unsimplified SVG vertices."""
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from reports.battery_power_exp004 import REPORT, ROOT, metrics, sha


def verify():
    manifest=json.loads((REPORT/'evidence/battery_power_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sources'].items():
        assert sha(ROOT/name)==digest, name
    for name,digest in manifest['outputs'].items():
        assert sha(REPORT/name)==digest, name
    full=pd.read_csv(REPORT/'evidence/battery_power_full_year.csv.gz')
    assert len(full)==4*52560 and not full.duplicated(['variant','interval_end']).any()
    stats=pd.read_csv(REPORT/'evidence/battery_power_metrics.csv')
    for key,part in full.groupby('variant',sort=False):
        time=pd.to_datetime(part.interval_end)
        assert time.iloc[0]==pd.Timestamp('2025-01-01 00:10')
        assert time.iloc[-1]==pd.Timestamp('2026-01-01 00:00')
        assert time.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all()
        np.testing.assert_allclose(part.net_charge_kw,part.charge_kw-part.discharge_kw,atol=1e-8)
        np.testing.assert_allclose(part.delta_power_kw.iloc[1:],np.diff(part.net_charge_kw),atol=1e-8)
        p=part[part.phase=='evaluation'].net_charge_kw.to_numpy()
        row=stats[(stats.variant==key)&(stats.scope=='evaluation')].iloc[0]
        np.testing.assert_allclose(row.total_variation_kw,np.abs(p[1:]-p[:-1]).sum())
        np.testing.assert_allclose(row.mean_abs_delta_kw,row.total_variation_kw/48095)
    for name,count,length in [('full-year',4,52560),('random-days',16,144)]:
        tree=ET.parse(REPORT/f'figures/battery-power-{name}.svg')
        lines=tree.findall('.//{http://www.w3.org/2000/svg}polyline')
        assert len(lines)==count
        assert all(len(line.attrib['points'].split())==length for line in lines)
    # Known independent example: [+1000,-1000,0,+1000] -> [-2000,+1000,+1000].
    simple=metrics([1000,-1000,0,1000])
    assert simple['total_variation_kw']==4000 and simple['direct_reversals']==1
    np.testing.assert_allclose(simple['rms_delta_kw'],np.sqrt(2000000))
    assert simple['max_abs_delta_kw']==2000
    page=(REPORT/'report.html').read_text(encoding='utf-8')
    assert page.count('<!-- battery-power:start -->')==1
    assert '电池充放电功率核验' in page
    build=json.loads((REPORT/'report_build.json').read_text(encoding='utf-8'))
    assert build['html_sha256']==sha(REPORT/'report.html')
    result={'status':'passed','rows':len(full),'checks':['source/output hashes','full-year timestamp coverage','CSV power and differences','evaluation total variation and denominator','all SVG vertices retained','known signed-power example','single embedded appendix and report hash'],
            'browser_visual_check':'not_performed: local file navigation rejected by browser security policy',
            'command':'python -m reports.verify_battery_power_exp004'}
    (REPORT/'evidence/battery_power_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    verify()
