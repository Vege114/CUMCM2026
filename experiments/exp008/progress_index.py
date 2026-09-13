"""Rebuild an auditable development index; never an achieved-goal report.

Only complete, same-source 334-day Q2 archives enter the annual target gate.
Short pilots remain separately listed and are not compared to annual totals.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import verify_npz, BASELINE_COST

OUT=Path('data/results/exp008')
INITIAL={'2':1421.7991105135516,'3':1507.622628752613,
         '4-2':1390.382746315672,'4-3':1522.7387288211346}


def main():
    rows=[]
    reports={}
    for path in sorted(OUT.rglob('dispatch*.npz')):
        if not any((path.parent/name).exists() for name in ('summary.json','completion.json')):
            continue
        scenario=path.stem.removeprefix('dispatch_')
        if scenario == 'dispatch': scenario='2'
        if scenario not in INITIAL: continue
        with np.load(path) as archive:
            if not {'fees','original','states'}.issubset(archive.files): continue
            days=len(archive['original'])
            if days>334: continue
        audit=path.parent/'planning_audit.json'
        if not audit.exists(): audit=path.parent/'audit.json'
        report=verify_npz(path,scenario,expected_days=days,start_day=31,
                          initial_soc=INITIAL[scenario],audit_path=audit if audit.exists() else None)
        run=str(path.parent.relative_to(OUT))
        correction_path=path.parent/'correction.json'
        correction=json.loads(correction_path.read_text()) if correction_path.exists() else {}
        selection_eligible=correction.get('eligible_for_selection',True)
        if not selection_eligible:
            report['selection_exclusion']=correction
        reports[run]=report
        b=report.get('battery_metrics',{})
        bill=report.get('billing',{})
        row=dict(run=run,scenario=scenario,days=days,passed=report['passed'],
                 full_comparable_q2=scenario=='2' and days==334 and report['passed'] and selection_eligible,
                 eligible_for_selection=selection_eligible,
                 total_cost=bill.get('total_cost'),
                 planned_cost=bill.get('planned_cost'),
                 up_cost=bill.get('up_cost'),down_cost=bill.get('down_cost'),
                 emergency_cost=bill.get('emergency_cost'),
                 direction_reversals=b.get('direction_reversals'),
                 active_slots=b.get('active_slots'),throughput_kwh=b.get('throughput_kwh'),
                 equivalent_full_cycles=b.get('equivalent_full_cycles'),
                 power_ramp_total_kw=b.get('power_ramp_total_kw'),
                 simultaneous_slots=b.get('simultaneous_slots'),
                 realized_physics_only=True,
                 known_initializer_emergency_charging_relaxation='mode_planning' in path.parts,
                 goal_passed=selection_eligible and report.get('goal',{}).get('passed',False),
                 archive=str(path),archive_sha256=report['source_sha256'])
        row['cost_reduction_vs_exp006_pct']=(100*(1-row['total_cost']/BASELINE_COST)
                                          if row['full_comparable_q2'] else None)
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'development_index.csv',index=False)
    annual=frame[frame.full_comparable_q2].sort_values('total_cost')
    low_count=annual[annual.direction_reversals<2729]
    stricter=low_count[(low_count.active_slots<23028)&(low_count.throughput_kwh<11480847.039823763)]
    status=dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                status='target_achieved_needs_final_deliverables' if annual.goal_passed.any() else 'target_not_achieved',
                evaluation_regime='development on an already examined year; not an untouched test',
                all_completed_archives=len(frame),complete_q2_archives=len(annual),
                physically_failed_archives=frame[~frame.passed]['run'].tolist(),
                best_cost=annual.head(1).to_dict('records'),
                best_cost_with_fewer_reversals=low_count.head(1).to_dict('records'),
                best_cost_with_all_intensity_metrics_lower=stricter.head(1).to_dict('records'),
                achieved_q2_runs=annual[annual.goal_passed]['run'].tolist(),
                fixed_baseline_cost_yuan=BASELINE_COST,
                fixed_eight_percent_target_yuan=.92*BASELINE_COST,
                final_report_written=False)
    (OUT/'development_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    (OUT/'development_verification.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
