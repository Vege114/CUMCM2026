"""Read-only verification and evidence extraction for the accepted exp008 selection.

No forecasting, model fitting, planning, or policy replay is run. Existing arrays
are checked against raw CSVs and existing measured audit fields are summarized.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from experiments.exp008.verify import verify_npz, verify_arrays, source_arrays
from experiments.exp008.q1 import audit as audit_q1
from experiments.exp008.frozen_sources import exp005_registry, final_template

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/results/exp008'
OUT = ROOT / 'reports/experiments/exp008/evidence'
Q2 = BASE / 'mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
UP = BASE / 'absolute_hgb_update_dispatch'
Q4 = BASE / 'q4_hgb_linked_price_physical'
SPECS = {
 '2': {'directory': Q2, 'archive': 'dispatch.npz', 'audit': 'planning_audit.json',
       'warmup': 'data/results/exp003/warmup_2.npz',
       'forecast': 'hgb_extra_trees_half_ridge28_memory',
       'controller': 'three-path physical MIP; ten-minute mask, hold1, planned daily cap8; fixed-mask full28-history L-BFGS-B refinement; causal strict-mask greedy execution',
       'forecast_difference': 'raw HGB/ExtraTrees average 0.5/0.5, joint Ridge28 on that raw blend, nonrecursive previous-day underlying load-error memory gain0.5',
       'extra_audits': [Q2/'independent_audit.json', Q2/'complete_provenance.json', Q2/'matched_HGB_hold1_effect_summary.json', Q2/'numerical_invocation_comparison.json'],
       'causality': [BASE/'forecast_hgb_extra_trees_half/causality_verification.json', BASE/'forecast_absolute_hgb/causality_verification.json', BASE/'forecast_absolute_extra_trees/causality_verification.json']},
 '3': {'directory': UP/'full334/3', 'archive': 'dispatch_3.npz', 'audit': 'audit.json',
       'warmup': 'data/results/exp002/warmup_3.npz', 'forecast': 'direct_hgb_ridge28_memory_half + legal official PV/current-prefix intraday updates',
       'controller': 'seven-path joint LP surrogate with next-update opportunity weight1.5; greedy execution with20kWh charging deadband; updates0/6/12/18; no daily switching cap',
       'forecast_difference': 'single absolute HGB midnight forecast, then Ridge28 and nonrecursive load memory; later issued official PV replaces midnight PV; load correction uses observed prefix',
       'extra_audits': [UP/'full334/3/paired_findings.json', UP/'full334/3/completion.json', UP/'protocol.json'],
       'causality': [UP/'causality_verification.json', BASE/'forecast_absolute_hgb/causality_verification.json']},
 '4-2': {'directory': Q4/'full334', 'archive': 'dispatch_4-2.npz', 'audit': 'audit.json',
       'warmup': 'data/results/exp002/warmup_4-2.npz', 'forecast': 'direct_hgb_ridge28_memory_half + Ridge28_plus_own_issued_HGB_load_PV_price',
       'controller': 'three-path physical hourly-mask MIP; switching50/wear0.002; fixed-mask full28-history refinement; causal strict-mask greedy execution',
       'forecast_difference': 'single HGB load/PV unchanged; price Ridge adds two own-issued load/PV regressors scaled by1000kW; historical rows use their own historical midnight issue',
       'extra_audits': [Q4/'full334/independent_audit.json', Q4/'controlled_input_audit.json', Q4/'source_verification.json', Q4/'three_layer_independent_verification.json', Q4/'metadata_erratum.json', Q4/'full334/source_corrected_audit.json'],
       'causality': [Q4/'price_causality_verification.json', BASE/'hgb_linked_price_forecast/causality_audit.json', BASE/'hgb_linked_price_forecast/independent_audit.json', BASE/'forecast_absolute_hgb/causality_verification.json']},
 '4-3': {'directory': UP/'full334/4-3', 'archive': 'dispatch_4-3.npz', 'audit': 'audit.json',
       'warmup': 'data/results/exp002/warmup_4-3.npz', 'forecast': 'direct_hgb_ridge28_memory_half + legal intraday PV/load updates + original causal price Ridge28',
       'controller': 'seven-path joint LP surrogate with next-update opportunity weight1.5; greedy execution with20kWh charging deadband; updates0/6/12/18; no daily switching cap',
       'forecast_difference': 'single HGB load/PV chain plus legal update; price model is original causal Ridge28, not Q4-2 new linked price',
       'extra_audits': [UP/'full334/4-3/paired_findings.json', UP/'full334/4-3/completion.json', UP/'protocol.json'],
       'causality': [UP/'causality_verification.json', BASE/'forecast_absolute_hgb/causality_verification.json']},
}


def read(p):
 return json.loads(Path(p).read_text())


def rel(p):
 p=Path(p)
 return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def ref(p):
 p=Path(p)
 if not p.is_absolute(): p=ROOT/p
 return {'path':rel(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}


def write(name, value):
 OUT.mkdir(parents=True, exist_ok=True)
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def independent_equations(a, scenario):
 """A second calculation does not call the shared verification equations."""
 eta=np.sqrt(.9)
 c,d,e,w = (a[k] for k in ['charge','discharge','emergency','surplus'])
 rawL=pd.read_csv(ROOT/'data/raw/附件2_小区负载.csv').iloc[31:,1:].to_numpy(float)
 rawPV=pd.read_csv(ROOT/'data/raw/附件2_光伏发电实际功率.csv').iloc[31:,1:].to_numpy(float)
 prices=(pd.read_csv(ROOT/'data/raw/附件4.csv').iloc[31:,1:].to_numpy(float) if scenario.startswith('4')
         else np.broadcast_to(pd.read_csv(ROOT/'data/raw/附件1.csv').iloc[:,1].to_numpy(float),c.shape))
 fee=np.stack([prices*a['original'],1.5*prices*np.maximum(a['final']-a['original'],0),
               .5*prices*np.maximum(a['original']-a['final'],0),5*prices*e],axis=-1)
 mode=np.where(c>1e-6,1,np.where(d>1e-6,-1,0)).ravel()
 active=mode[mode!=0]
 metrics={
  'raw_load_max_difference_kw':float(np.max(abs(a['actual'][:,:,0]-rawL))),
  'raw_pv_max_difference_kw':float(np.max(abs(a['actual'][:,:,1]-rawPV))),
  'raw_settlement_price_max_difference':float(np.max(abs(a['price']-prices))),
  'max_balance_error_kwh':float(np.max(abs(a['final']+e+d+rawPV/6-rawL/6-c-w))),
  'max_soc_equation_error_kwh':float(np.max(abs(np.diff(a['states'],axis=1)-(eta*c-d/eta)))),
  'max_midnight_soc_error_kwh':float(np.max(abs(a['states'][1:,0]-a['states'][:-1,-1]))),
  'max_archived_fee_error_yuan':float(np.max(abs(fee-a['fees']))),
  'recomputed_fee_components_yuan':dict(zip(['planned','up','down','emergency'],map(float,fee.sum(axis=(0,1))))),
  'recomputed_total_yuan':float(fee.sum()),'simultaneous_slots':int(np.sum((c>1e-6)&(d>1e-6))),
  'emergency_while_charging_slots':int(np.sum((e>1e-6)&(c>1e-6))),
  'emergency_total_kwh':float(e.sum()),
  'nonidle_direction_reversals':int(np.sum(active[1:]!=active[:-1])),
  'direct_adjacent_sign_reversals':int(np.sum(mode[1:]*mode[:-1]==-1)),
  'charge_episodes':int(np.sum((mode==1)&(np.r_[0,mode[:-1]]!=1))),
  'discharge_episodes':int(np.sum((mode==-1)&(np.r_[0,mode[:-1]]!=-1))),
  'active_slots':int(np.sum(mode!=0)),
 }
 metrics['total_episodes']=metrics['charge_episodes']+metrics['discharge_episodes']
 metrics['passed']=all(v<1e-6 for k,v in metrics.items() if k.startswith(('raw_','max_'))) and metrics['simultaneous_slots']==0 and metrics['emergency_while_charging_slots']==0
 return metrics


def audit_selected():
 cases={}
 for s,sp in SPECS.items():
  p=sp['directory']/sp['archive']
  with np.load(ROOT/sp['warmup']) as z:
   soc=float(z['states'][-1,-1]); pw=float(6*(z['charge'][-1,-1]-z['discharge'][-1,-1]))
   modes=np.sign((z['charge']-z['discharge']).ravel()); nz=modes[np.abs(modes)>0]
   last_nonidle=int(nz[-1]) if len(nz) else 0
  v=verify_npz(p,s,audit_path=sp['directory']/sp['audit'],initial_soc=soc,initial_power_kw=pw,initial_mode=int(np.sign(pw)))
  with np.load(p) as z: a={k:z[k].copy() for k in z.files}
  direct=independent_equations(a,s)
  records=read(sp['directory']/sp['audit'])
  origins=[r['day']*144+int(r.get('slot',0)) for r in records]
  expected=[day*144+slot for day in range(31,365) for slot in ([0,36,72,108] if s in ['3','4-3'] else [0])]
  issue_checks={'actual_issue_count':len(origins),'expected_issue_count':len(expected),'exact_issue_grid':origins==expected,
   'forecast_information_cutoff_matches_issue':all(r['forecast']['information_cutoff_exclusive']==o for r,o in zip(records,origins)),
   'max_observed_index_before_issue':all(r['forecast']['max_observed_index']<o for r,o in zip(records,origins)),
   'future_realized_price_not_declared_known':all(not r['forecast']['known_future_price'] for r in records),
   'scope':'all archived issue metadata; independent mutation evidence below is separate and sampled, not an exhaustive proof'}
  cases[s]={'archive':ref(p),'issue_audit':ref(sp['directory']/sp['audit']), 'warmup':ref(ROOT/sp['warmup']),
   'evaluation_initial_soc_kwh':soc,'evaluation_initial_power_kw':pw,'evaluation_initial_mode':int(np.sign(pw)),
   'warmup_last_nonidle_mode':last_nonidle,'warmup_to_formal_soc_difference_kwh':float(a['states'][0,0]-soc),
   'forecast_model':sp['forecast'],'controller':sp['controller'],'cross_question_difference':sp['forecast_difference'],
   'verification':v,'independent_direct_equations':direct,'issue_grid_checks':issue_checks,
   'available_additional_audits':[ref(x) for x in sp['extra_audits']], 'available_causality_evidence':[ref(x) for x in sp['causality']],
   'physical_and_billing_ready':v['passed'] and direct['passed'] and all(issue_checks[k] for k in ['exact_issue_grid','forecast_information_cutoff_matches_issue','max_observed_index_before_issue','future_realized_price_not_declared_known'])}
  assert cases[s]['physical_and_billing_ready'],s
 q1p=BASE/'q1/revised.json';q1=read(q1p); q={k:np.asarray(v,float) for k,v in q1['trajectory'].items()}
 actual,price=source_arrays('1',0,1)
 a=dict(original=q['g'][None],final=q['g'][None],charge=q['c'][None]/6,discharge=q['d'][None]/6,
  emergency=np.zeros((1,144)),surplus=q['w'][None],states=q['E'][None],actual=actual,price=price[None],
  fees=np.stack([q['g'][None]*price,np.zeros((1,144)),np.zeros((1,144)),np.zeros((1,144))],axis=-1))
 v=verify_arrays(a,'1',expected_days=1,start_day=0,initial_soc=6000,initial_mode=0,initial_power_kw=0,source_actual=actual,source_price=price)
 specialized=audit_q1(np.column_stack((price,actual[0])),dict(q1,trajectory=q))
 cases['1']={'archive':ref(q1p),'forecast_model':'deterministic attachment1 given day; no learned forecast',
  'controller':'main two-stage mixed-integer cost-first then normalized operational smoothness; delta0.001',
  'evaluation_initial_soc_kwh':6000.,'evaluation_initial_power_kw':0.,'evaluation_initial_mode':0,
  'initial_power_note':'zero is metric reference; Q1 cyclic operation has end-to-start constraints',
  'source_units':{'g':'kWh','c':'kW','d':'kW','E':'kWh','w':'kWh'},'conversion':'charge/discharge kWh = c/d divided by6; purchase g already kWh',
  'issue_audit':None,'available_causality_evidence':[],'causality_not_applicable_reason':'all attachment1 given before deterministic optimization',
  'stages':q1['stages'],'verification':v,'specialized_two_stage_audit':specialized,'physical_and_billing_ready':v['passed'] and specialized['violations']==0}
 templates=[]
 for f in ['report.md','history-comparison.md','battery-power.md']:
  name='reports/templates/'+f; data, provenance=final_template(name,root=ROOT)
  templates.append(provenance)
 old=read(BASE/'baselines.json')
 historical, historical_source=exp005_registry(ROOT)
 result={'review_scope':'accepted selected archives and original requirements; no new experiment or refitting',
  'assessment':'Share with caveats: all selected physics and original actual settlement verified; final model equality across questions is not claimed',
  'reviewer':'independent audit subagent; second raw-CSV equation calculation plus existing verification',
  'evidence_builder':ref(Path(__file__)),'report_templates_from_main':templates,
  'problem_sources':[ref(ROOT/'C题/mineru-raw/C题.md'),ref(ROOT/'C题/C题.md'),ref(ROOT/'C题/requirements.md'),ref(ROOT/'C题/conversion.md')],
  'formal_period':{'start':'2025-02-01','end':'2025-12-31','days':334,'intervals_per_scenario':48096,'warmup':'January archived causal policies initialize each scenario; not the final HGB/blend trained policy'},
  'shared_physics':{'capacity_kwh':12000,'soc_bounds_kwh':[1200,10800],'power_bound_kw':5000,'dt_hours':1/6,'eta_rt_assumption':.9,'eta_one_way':float(np.sqrt(.9))},
  'scenarios':cases,'baseline':old['primary_comparator'],
  'acceptance':{'user_message_relayed_by_parent':'行，我觉得现在这个结果比较满意了，就这样写报告然后提交吧','reason':'user accepts current verified result; further optimization stopped','original_8pct_goal_passed':False,'original_10pct_goal_passed':False,'actual_q2_cost_reduction_pct':cases['2']['verification']['goal']['cost_reduction_pct'],'accepted_archive_sha256':cases['2']['archive']['sha256']},
  'required_disclosures':[
   'Q1 deterministic two-layer solution; Q2 blend, Q3/Q4 single HGB; Q4-3 does not use Q4-2 new linked price. Four questions share physical/settlement definitions, not one final prediction model.',
   'Q2 cost reduced6.1443%, below original8%-10%; archived goal failure remains true; user explicitly accepted current result.',
   'Q2 nonidle reversals2533 vs2729 and throughput decreased; charge+discharge episodes3898 vs3791 and active slots37211 vs23028 increased. Never redefine all operation counts as reversals.',
   'Q2 daily cap8 constrains planned binary mode switches; actual nonidle direction and planned mode differ. No cap8 or Q1 ramp/hold constraints are guaranteed in Q3/Q4-3.',
   'Battery throughput/EFC/TV are operational proxies, not identified ageing model or lifetime claims; etaRT.9 is a team assumption.',
   'Daily MIPs have bounded gaps/time limits; scenario recourse lacks a full nonanticipativity certificate; L-BFGS-B refinement often hits iteration budget. Feasibility and actual causal execution do not establish global optimality.',
   'The334 days are repeatedly examined chronological development evaluation, not an untouched independent test; final model only seed42, no invented across-seed variance.',
   'Q2→Q3 or Q4-2→Q4-3 fee differences mix forecast/controller/price differences and cannot isolate value of six-hour updates.',
   'Final Q3/Q4-3 settlement charges original purchases and final net adjustment once; proposal fees are not repeatedly added; artificial penalties/terminal values excluded from bills.',
   'Q4-2 corrected metadata view supersedes stale inherited load_method/comparison_scope prose; original signed numerical archive unchanged.',
   'Q1 stage1 cost34080.95139956883 is the minimum within added ramp/hold constraints; final stage2 cost34115.032450968385. Do not claim globally unconstrained minimum.',
  ],
  'exp005_registry':{**historical_source,'read_and_verified':True},
  'remaining_delivery_checks':['report mathematical/fee/claim review after draft exists','five template workbooks cell/readback verification','specified four-date tables and emergency interval reconciliation','main-template history/forecast/runtime/battery plots; raw144-point random days and full48096 CSV','final report HTML/PDF rendering if produced; manifest/hashes/commit status'],
  'new_optimization_blockers':[], 'work_remains_is_delivery_not_model_search':True}
 write('final_selection_audit.json',result)
 return result


def stats(values):
 v=np.asarray([x for x in values if x is not None],float)
 return {'count':int(v.size),'sum':float(v.sum()) if v.size else None,'mean':float(v.mean()) if v.size else None,'median':float(np.median(v)) if v.size else None,'p95':float(np.quantile(v,.95)) if v.size else None,'max':float(v.max()) if v.size else None}


def runtime_and_solvers():
 runtime={'unit':'seconds','unknown_value':None,'scope':'existing measured fields only; no durations inferred from timestamps; stages can overlap and must not be added to reported wall time', 'components':[]}
 solver={'scope':'selected final runs; gaps describe surrogate solve only, not final global optimality','scenarios':{}}
 for family in ['forecast_absolute_hgb','forecast_absolute_extra_trees']:
  p=BASE/family/'training_audit.json'; rows=read(p); models=[m for row in rows for m in row['models']]
  runtime['components'].append({'id':family,'stage':'model_fit_and_validation_prediction','source':ref(p),'field':'[*].models[*].training_seconds','model_count':len(models),'seed':42,'fit_plus_validation_prediction_seconds':stats([m['training_seconds'] for m in models]),'pure_fit_seconds':None,'validation_prediction_seconds':None,'feature_construction_seconds':None,'prediction_seconds':None,'calibration_seconds':None,'note':'archived training_seconds begins immediately before model.fit and ends after the subsequent validation-set model.predict; pure fit and validation prediction cannot be separated. Formal-horizon inference is outside this field. Same fitted models reused by final selection: count once, not once per question; no end-to-end timing claim'})
 q1=read(BASE/'q1/revised.json')
 runtime['components'].append({'id':'Q1','stage':'two_stage_optimization','source':ref(BASE/'q1/revised.json'),'field':'stages[*].seconds','stages':q1['stages'],'sum_solver_seconds':sum(r['seconds'] for r in q1['stages']),'export_seconds':None,'report_seconds':None})
 solver['scenarios']['1']={'stages':q1['stages'],'formulation':'mixed integer; stage2 constrained to1.001 times stage1 operational feasible cost','relevant_configuration':q1['config']}
 for s,sp in SPECS.items():
  p=sp['directory']/sp['audit']; rows=read(p)
  key='mip' if s in ['2','4-2'] else 'solver'
  solved=[r[key] for r in rows]; refinements=[r.get('greedy_refinement',r.get('refinement')) for r in rows]; refinements=[x for x in refinements if x is not None]
  wall=read(sp['directory']/'completion.json').get('wall_seconds') if (sp['directory']/'completion.json').exists() else None
  runtime['components'].append({'id':'Q'+s,'stage':'dispatch_run','source':ref(p),'solver_seconds':stats([x.get('seconds') for x in solved]),'planning_envelope_seconds':stats([x.get('planning_seconds') for x in solved]),'refinement_seconds':stats([x.get('planning_seconds') for x in refinements]),'run_wall_seconds':wall,'wall_source':ref(sp['directory']/'completion.json') if wall is not None else None,'prediction_seconds':None,'calibration_seconds':None,'actual_execution_seconds':None,'audit_seconds':None,'export_seconds':None,'report_seconds':None,'note':'planning_envelope includes solver where logged; run wall may include forecasts/planning/execution/I/O; unavailable substage timings are null'})
  gaps=[x.get('mip_gap') for x in solved]
  solver['scenarios'][s]={'source':ref(p),'solve_count':len(solved),'methods':dict(Counter(x.get('method') for x in solved)), 'status_counts':dict(Counter(str(x.get('status')) for x in solved)), 'feasible_count':sum(x.get('feasible',False) for x in solved),'gap':stats(gaps),'gap_above_0_005':sum(x is not None and x>.005 for x in gaps),'gap_above_0_01':sum(x is not None and x>.01 for x in gaps),'constraint_residual':stats([x.get('constraint_residual') for x in solved]),'nodes':stats([x.get('node_count') for x in solved]),'variables':stats([x.get('variables') for x in solved]),'binaries':stats([x.get('binaries') for x in solved]),'constraints':stats([x.get('constraints') for x in solved]),'refinement':{'attempts':len(refinements),'successes':sum(x['success'] for x in refinements),'messages':dict(Counter(x['message'] for x in refinements)),'iterations':stats([x['iterations'] for x in refinements]),'evaluations':stats([x['evaluations'] for x in refinements]),'all_global_optimality_certificate':False},'full_nonanticipative_recourse_certificate':False,'actual_causal_execution_and_physics_verified_separately':True}
 solver['scenarios']['2']['configuration']=read(Q2/'evaluation_protocol.json')
 solver['scenarios']['4-2']['configuration']=read(Q4/'protocol.json')['config']
 for s in ['3','4-3']:solver['scenarios'][s]['configuration']=read(SPECS[s]['directory']/'completion.json')['config']
 runtime['total_end_to_end_seconds']=None
 write('runtime_by_stage.json',runtime);write('planning_solver_diagnostics.json',solver)


def raw_validation():
 rows=[]
 expected=pd.date_range('2025-01-01','2025-12-31')
 def slot_minutes(x):
  x=str(x).strip()
  if '+1' in x:return 1440
  parts=x.split(':');return int(parts[0])*60+int(parts[1])
 for p in sorted((ROOT/'data/raw').glob('*.csv')):
  df=pd.read_csv(p);row={'source':ref(p),'shape':list(df.shape),'column_count':len(df.columns),'duplicate_columns':int(df.columns.duplicated().sum()),'original_null_cells':int(df.isna().sum().sum())}
  if p.name=='附件3.csv':
   ff=df.copy();ff.iloc[:,0]=ff.iloc[:,0].ffill();dates=pd.to_datetime(ff.iloc[:,0]);hour=ff.iloc[:,1].map(lambda t:int(str(t).split(':')[0])); keys=list(zip(dates.astype(str),hour));num=ff.iloc[:,2:].to_numpy(float)
   row.update({'date_forward_fill_only_for_merged_date_cells':True,'null_dates_before_ffill':int(df.iloc[:,0].isna().sum()),'null_cells_after_date_ffill':int(ff.isna().sum().sum()),'release_count':len(keys),'unique_release_count':len(set(keys)),'each_day_releases':[0,6,12,18],'all_days_exact_four_releases':all(sorted(hour[dates==d].tolist())==[0,6,12,18] for d in expected),'lead_hours':list(range(1,25)),'value_semantics':'24 hourly forecast knots after issue; linear interpolation plus interval integration; observed preceding10minute anchor is legal','future_issue_access_rule':'release origin <= current issue; future releases forbidden'})
  elif p.name=='附件1.csv':
   mins=[slot_minutes(x) for x in df.iloc[:,0]];num=df.iloc[:,1:].to_numpy(float)
   row.update({'exact_right_end_grid_10_to_1440_minutes':mins==list(range(10,1441,10)),'known_before_operation':True,'column_units':['time_right_endpoint','yuan/kWh','kW','kW']})
  else:
   dates=pd.to_datetime(df.iloc[:,0]);num=df.iloc[:,1:].to_numpy(float);mins=[slot_minutes(x) for x in df.columns[1:]]
   row.update({'dates_exact2025':pd.DatetimeIndex(dates).equals(expected),'duplicate_dates':int(dates.duplicated().sum()),'exact_right_end_grid_10_to_1440_minutes':mins==list(range(10,1441,10)),'chronological_intervals':int(num.size),'value_unit':'yuan/kWh' if p.name=='附件4.csv' else 'kW','formal_slice_rows':[31,365]})
  row.update({'numeric_finite':bool(np.isfinite(num).all()),'numeric_missing_count':int(np.isnan(num).sum()),'numeric_negative_count':int(np.sum(num<0)),'numeric_min':float(num.min()),'numeric_max':float(num.max())});rows.append(row)
 result={'scope':'raw CSV schema, original bytes, dates, right-end10minute time semantics and final selected archive source equality','source_is_task_provided_not_live_data':True,'files':rows,'unit_contract':{'raw_load_PV':'interval kW; planning energy kWh=value/6','charge_discharge_annual_archives':'AC kWh/10min; plotted signed power =6*(charge-discharge)','soc':'kWh;145 boundaries/day','price':'yuan/kWh','fees':'yuan; four components original/up/down/emergency'},'time_semantics':{'timezone':'source local calendar; no timezone conversion','interval':'row endpoint00:10 corresponds[00:00,00:10); 0:00+1 means24:00 end of same row day','issue_origin_index':'zero-based day*144+slot; observed indices strictly beloworigin','formal_start_index':4464,'formal_stop_index_exclusive':52560,'formal_days':334,'formal_intervals':48096},'train_validation_protocol':{family:ref(BASE/family/'protocol.json') for family in ['forecast_absolute_hgb','forecast_absolute_extra_trees','forecast_hgb_extra_trees_half']},'no_future_label_claim_scope':'full stored cutoff audit plus sampled future-mutation audits, see final_selection_audit.json','raw_data_checks_passed':all(r['numeric_finite'] and r['numeric_negative_count']==0 and r['duplicate_columns']==0 and r.get('dates_exact2025',True) and r.get('exact_right_end_grid_10_to_1440_minutes',True) and r.get('all_days_exact_four_releases',True) for r in rows)}
 assert result['raw_data_checks_passed']
 write('data_validation.json',result)


def seeds_candidates():
 selected=[]
 for family in ['forecast_absolute_hgb','forecast_absolute_extra_trees']:
  protocol=read(BASE/family/'protocol.json'); models=read(BASE/family/'training_audit.json')
  selected.append({'family':family,'protocol':ref(BASE/family/'protocol.json'),'seeds':[protocol['model_configuration']['random_state']],'monthly_model_count':sum(len(r['models']) for r in models),'complete_months':[r['month'] for r in models],'cross_seed_n':1,'cross_seed_mean':None,'cross_seed_standard_deviation':None,'interpretation':'22 monthly channel fits share one seed; months are not independent seed repetitions'})
 records=[]
 excluded={'source_archive','source_snapshots','forecast_archive','forecast_evidence','source'}
 for p in sorted(BASE.rglob('summary.json')):
  relative=p.relative_to(BASE)
  if any(x in excluded or x.startswith('source_archive') for x in relative.parts):continue
  d=read(p)
  if not isinstance(d,dict):continue
  entry={'candidate_path':str(relative.parent),'source':ref(p),'completed_days':d.get('completed_days',d.get('days')), 'explicit_complete':d.get('complete'),'total_cost_yuan':d.get('total_cost',d.get('bridge_total_cost')),'formal_selected':any(p.parent==sp['directory'] for sp in SPECS.values()),'automatic_full_year_ranking':False}
  for k in ['predictive_gate_passed','metric_checks','gate_checks','cost_reduction_pct','verified','goal_eligible','final_model_selected','final_model_selection','full_physical_performed','bridge_performed','partial','error','failure']:
   if k in d:entry[k]=d[k]
  for key in ['verification','goal','actual','sum']:
   v=d.get(key)
   if isinstance(v,dict):
    entry[key+'_reported_subset']={k:v[k] for k in ['passed','days','total_cost','total_cost_yuan','cost_change','cost_change_yuan','recomputed_total_cost','cost_reduction_pct'] if k in v}
    if key=='verification' and isinstance(v.get('goal'),dict):entry['original_8pct_gate']=v['goal']
  records.append(entry)
 failure=BASE/'forecast_absolute_extra_trees_initial_audit_failure/failure_explanation.json'
 explicit_failure={'source':ref(failure),'explanation':read(failure)}
 controls=[]
 for p in [Q2/'matched_HGB_hold1_effect_summary.json', Q2/'numerical_invocation_comparison.json', Q2/'paired_findings.json',Q4/'comparison.json',Q4/'controlled_input_audit.json',UP/'full334/3/paired_findings.json',UP/'full334/4-3/paired_findings.json',BASE/'forecast_ensemble_diagnostic/summary.json',BASE/'multistart_refinement/summary.json',BASE/'mode_refinement_diagnostic/paired_5days/summary.json',BASE/'cumulative_risk_supplement_v1/supplementary_audit.json']:
  if p.exists():controls.append(ref(p))
 write('seed_and_ablation_results.json',{'scope':'selected-family seed audit plus discovered existing summary records and explicit failed audit; no experiment is rerun','selection_on_examined_development_year':True,'unexamined_test_year':None,'no_invented_seed_statistics':True,'selected_model_families':selected,'blend':{'weights':[.5,.5],'same_all334days':True,'weight_tuning':False,'source':ref(BASE/'forecast_hgb_extra_trees_half/protocol.json')},'diagnostic_random_day_seed':20260912,'diagnostic_seed_is_not_training_seed':True,'controlled_comparison_sources':controls,'summary_inventory_count':len(records),'summary_inventory':records,'explicit_failures':[explicit_failure],'scope_limits':['inventory lists present top-level experimental summary.json files, excludes frozen duplicate source/forecast copies','A failed8% gate is not solver infeasibility; a partial diagnostic is not a334-day competing policy','Different architecture, calibration, controller, horizon or tariff variants are not a seed ablation','After user acceptance, unfinished budget-feedback work was stopped and is not selected; no completion or benefit inferred from absent summary','Historical final_model_selected:false flags describe original candidate phase; accepted final selection is separate and hash bound']})


def main():
 result=audit_selected();runtime_and_solvers();raw_validation();seeds_candidates()
 print(json.dumps({'all_selected_cases_passed':all(c['physical_and_billing_ready'] for c in result['scenarios'].values()),'outputs':[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('*.json'))]},ensure_ascii=False))


if __name__=='__main__':main()
