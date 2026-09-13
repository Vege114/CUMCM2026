"""Reconcile final workbook matrices directly to immutable dispatch archives."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(a,b):
    clock=lambda t:f'{t//6:02d}:{t%6*10:02d}'
    return f'{clock(a)}-{clock(b)}'
def run():
    path=ROOT/'reports/experiments/exp008/final_payload.json';payload=json.loads(path.read_text());rows=[]
    for scenario,sc in payload['scenarios'].items():
        source=Path(sc['archive']['path']);assert sha(source)==sc['archive']['sha256']
        with np.load(source) as z:a={k:z[k] for k in z.files}
        data=sc['workbook_data']
        for key in ('original','final'):np.testing.assert_allclose(data[key],a[key],rtol=0,atol=1e-9)
        fees=a['fees'].sum(axis=(1,2));np.testing.assert_allclose(data['fees'],fees,rtol=0,atol=1e-8)
        for day,date in enumerate(data['dates']):
            b=data['battery'][day*6:(day+1)*6]
            for j,r in enumerate(b):
                assert r[0]==(date if j==0 else None) and r[1]==interval(j*24,(j+1)*24)
                np.testing.assert_allclose(r[2:4],[a['charge'][day,j*24:(j+1)*24].sum(),a['discharge'][day,j*24:(j+1)*24].sum()],rtol=0,atol=1e-8)
                assert r[4]==('0:00' if j==0 else '24:00' if j==1 else None)
                if j<2:np.testing.assert_allclose(r[5],a['states'][day,0 if j==0 else -1],rtol=0,atol=1e-8)
                else:assert r[5] is None
        expected=[]
        for day,date in enumerate(data['dates']):
            active=a['emergency'][day]>1e-6;events=[];start=None
            for t in range(145):
                yes=t<144 and active[t]
                if yes and start is None:start=t
                if not yes and start is not None:
                    events.append([date if not events else None,interval(start,t),float(a['emergency'][day,start:t].sum())]);start=None
            expected.extend(events if events else [[date,'无',0.]])
        assert len(expected)==len(data['emergency'])
        for actual,target in zip(data['emergency'],expected):
            assert actual[:2]==target[:2];np.testing.assert_allclose(actual[2],target[2],rtol=0,atol=1e-8)
        rows.append(dict(scenario=scenario,source_sha256=sha(source),days=334,plan_slots=48096,battery_blocks=2004,
            emergency_event_rows=len(expected),cash_yuan=float(fees.sum()),repeated_fee_column_counted_once=True,passed=True))
    q1=payload['q1'];source=Path(q1['archive']['path']);q=json.loads(source.read_text())['trajectory']
    # Q1 uses mixed units by design: g is already kWh, while c/d are kW.
    np.testing.assert_allclose([r[1] for r in q1['workbook_data']['plan_rows']],np.array(q['g']),rtol=0,atol=1e-9)
    for b,r in enumerate(q1['workbook_data']['battery_rows']):
        np.testing.assert_allclose(r[1:3],[np.array(q[k])[24*b:24*(b+1)].sum()/6 for k in ('c','d')],rtol=0,atol=1e-8)
    report=dict(passed=True,payload_sha256=sha(path),source_audits=rows,Q1_kW_to_kWh_conversion_verified=True,
        audit_script_sha256=sha(Path(__file__)))
    (ROOT/'reports/experiments/exp008/evidence/workbooks/source_to_payload_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=True,annual_scenarios=len(rows),Q1_conversion=True)))
if __name__=='__main__':run()
