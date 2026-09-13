"""Rebind frozen report evidence to this checkout without changing accepted results."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from experiments.exp008.report_payload import build_payload, enforce_final

ROOT=Path(__file__).resolve().parents[2]


def digest(data):return hashlib.sha256(data).hexdigest()


def refresh():
    destination=ROOT/'reports/experiments/exp008/final_payload.json'
    previous=json.loads(destination.read_text()) if destination.exists() else None
    manifest=json.loads((ROOT/'experiments/exp008/final_selection.json').read_text())
    fresh=build_payload(manifest,mode='final');enforce_final(fresh)
    old_hash=digest(destination.read_bytes()) if previous else None
    unchanged={}
    for scenario in ['1','2','3','4-2','4-3']:
        current=fresh['q1'] if scenario=='1' else fresh['scenarios'][scenario]
        old=(previous['q1'] if scenario=='1' else previous['scenarios'][scenario]) if previous else None
        same=old is None or old['workbook_data']==current['workbook_data']
        if not same:raise ValueError(f'Refusing changed workbook payload for Q{scenario}; this command only refreshes metadata')
        data=json.dumps(current['workbook_data'],sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
        unchanged[scenario]={'exactly_equal_to_previous':True if old else None,'workbook_data_sha256':digest(data),'archive_sha256':current['archive']['sha256']}
    encoded=(json.dumps(fresh,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    temporary=destination.with_suffix('.json.refreshing')
    temporary.write_bytes(encoded);os.replace(temporary,destination)
    audit={'passed':True,'previous_payload_sha256':old_hash,'current_payload_sha256':digest(encoded),
           'metadata_only':previous is not None,'workbook_data':unchanged,
           'no_training_or_optimization':True,'new_workbook_export_required':False if previous else None}
    (destination.parent/'evidence/payload_metadata_refresh.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':refresh()
