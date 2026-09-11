"""Resumable day-wise calibration, physical replay, and complete experiment comparisons."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .data import EPOCH,HERE,ROOT,SEEDS,Data,month_origins,protocol,signature
from .physics import deterministic,execute,settle,validate_detail
from .predict import ForecastStore,LegacyStore
from .risk import optimize,weighted_cvar
from .scenarios import ScenarioFactory

SCENARIOS=("2","3","4-2","4-3")


def day_run(data,store,day,scenario,initial,method="deterministic",weight=0,
            update_hours=(6,12,18),anticipate=True,raw_pv=False,known_price=False,
            factory=None,validation_month=None):
    origin=day*144;issued=scenario in ("3","4-3");variable=scenario.startswith("4")
    actual=data.actual[origin:origin+144].copy()
    price=actual[:,2].copy() if variable else data.fixed_price.copy()
    charge,discharge,emergency,surplus=(np.zeros(144) for _ in range(4))
    states=np.empty(145);states[0]=initial
    original=None;final=np.zeros(144);logs=[];first_tree=None
    boundaries=[0]+([h*6 for h in update_hours] if issued else [])+[144]
    began=time.monotonic()
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        o=origin+start
        if isinstance(store,LegacyStore):forecast=store.get(o,scenario)
        else:forecast=store.get(o,issued=issued,raw_pv=raw_pv,validation_month=validation_month)
        point=forecast[:144-start,[0,3 if issued else 1,2]].copy()
        if not variable or known_price:point[:,2]=price[start:]
        if method=="risk":
            bundle=factory.build(o,forecast,scenario,update_hours,raw_pv,anticipate)
            if known_price:bundle["paths"][:,:,2]=price[start:]
            plan,meta=optimize(bundle,states[start],point,weight=weight,
                               original=None if start==0 else original[start:],adjustable=issued)
            if first_tree is None:first_tree=bundle["metadata"]
            meta["source_latest_target"]=bundle["metadata"]["latest_source_target"]
            assert meta["source_latest_target"]<=o
        else:
            plan,meta=deterministic(point[:,0],point[:,1],point[:,2],states[start],
                                    None if start==0 else original[start:])
        meta.update(origin=o,day=day,scenario=scenario,information_cutoff=o,
                    issued_forecast_allowed=issued,known_future_price=known_price)
        logs.append(meta)
        if start==0:original=plan.copy()
        final[start:]=plan
        c,d,e,w,s=execute(final[start:stop],actual[start:stop,0],actual[start:stop,1],states[start])
        charge[start:stop]=c;discharge[start:stop]=d;emergency[start:stop]=e;surplus[start:stop]=w
        states[start:stop+1]=s
    fees=settle(original,final,emergency,price)
    detail={"original":original,"final":final,"charge":charge,"discharge":discharge,
            "emergency":emergency,"surplus":surplus,"states":states,"fees":fees,
            "price":price,"actual":actual}
    balance,state_error=validate_detail(detail)
    summary={"day":day,"date":str((EPOCH+pd.Timedelta(days=day)).date()),
        "month":int((EPOCH+pd.Timedelta(days=day)).month),"scenario":scenario,
        "planned_kwh":float(original.sum()),"final_kwh":float(final.sum()),
        "emergency_kwh":float(emergency.sum()),"emergency_minutes":int((emergency>1e-6).sum()*10),
        "charge_kwh":float(charge.sum()),"discharge_kwh":float(discharge.sum()),
        "surplus_kwh":float(surplus.sum()),"initial_soc":float(states[0]),"final_soc":float(states[-1]),
        "planned_cost":float(fees[:,0].sum()),"up_cost":float(fees[:,1].sum()),
        "down_cost":float(fees[:,2].sum()),"emergency_cost":float(fees[:,3].sum()),
        "total_cost":float(fees.sum()),"violations":0,"max_balance_error":balance,
        "max_state_error":state_error,"solve_execute_seconds":time.monotonic()-began,
        "solver_calls":len(logs),"timeout_count":sum(r["status"]==1 for r in logs),
        "fallback_count":sum(r["fallback"] for r in logs),
        "incumbent_count":sum(r.get("feasible_origin")=="fixed_controller_region_incumbent" for r in logs),
        "gap_certified_count":sum(r.get("mip_gap") is not None and r["mip_gap"]<=.010001 for r in logs),
        "weight":float(weight),"updates":len(boundaries)-2}
    return summary,detail,logs,first_tree


def evaluation_signature(data,run_id):
    training={p.name:json.loads(p.read_text())["signature"] for p in (HERE/"runs"/run_id).glob("m*.json")}
    return signature(("data.py","physics.py","risk.py","scenarios.py","predict.py","evaluate.py","protocol.json"),
                     {"data":data.hashes,"training":training,
                      "legacy_predictions":hashlib.sha256((ROOT/"data/results/exp001/ensemble_predictions.npz").read_bytes()).hexdigest()})


def warmup(data,scenario,run_id,sig):
    directory=HERE/"runs"/run_id/"warmup";directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{scenario}.npz";metadata=path.with_suffix(".json")
    if metadata.exists():
        old=json.loads(metadata.read_text());assert old["signature"]==sig
        with np.load(path) as f:details={k:f[k].copy() for k in f.files}
        return old["summaries"],details
    store=ForecastStore(data,run_id,kind="periodic");state=6000.;summaries=[];parts=[]
    for day in range(31):
        summary,detail,_,_=day_run(data,store,day,scenario,state)
        summaries.append(summary);parts.append(detail);state=summary["final_soc"]
    details={key:np.stack([p[key] for p in parts]) for key in parts[0]}
    np.savez_compressed(path,**details)
    metadata.write_text(json.dumps({"signature":sig,"summaries":summaries},indent=2))
    return summaries,details


def calibrate_one(arguments):
    run_id,scenario=arguments;data=Data();sig=evaluation_signature(data,run_id)
    out=HERE/"runs"/run_id/"calibration";out.mkdir(parents=True,exist_ok=True)
    path=out/f"{scenario}.json"
    if path.exists():
        result=json.loads(path.read_text());assert result["signature"]==sig;return result
    _,w=warmup(data,scenario,run_id,sig)
    store=ForecastStore(data,run_id,42);factory=ScenarioFactory(data,store)
    candidates=[]
    for weight in protocol()["risk"]["lambda_candidates"]:
        state=float(w["states"][24,0]);days=[]
        for day in range(24,31):
            summary,_,_,_=day_run(data,store,day,scenario,state,method="risk",weight=weight,
                                 factory=factory,validation_month=2)
            days.append(summary);state=summary["final_soc"]
        candidates.append({"weight":weight,"cost":sum(d["total_cost"] for d in days),"days":days})
        print("CALIBRATION",scenario,weight,candidates[-1]["cost"],flush=True)
    best=min(candidates,key=lambda r:(round(r["cost"],6),r["weight"]))
    result={"signature":sig,"scenario":scenario,"selected_weight":best["weight"],
            "selection_time":31*144,"validation_days":[24,30],"candidates":candidates,
            "selection_note":"January held-out predictions from the February checkpoint; scenario residuals remain prequential baseline errors"}
    path.write_text(json.dumps(result,indent=2));return result


def calibrate(run_id="exp002",workers=4):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(calibrate_one,[(run_id,s) for s in SCENARIOS]))
    out=ROOT/"data/results"/run_id;out.mkdir(parents=True,exist_ok=True)
    (out/"risk_calibration.json").write_text(json.dumps(results,indent=2))
    return {r["scenario"]:r["selected_weight"] for r in results}


def cases(weights):
    jobs=[]
    def add(name,scenario,**kw):
        job={"name":name,"scenario":scenario,"seed":42,"kind":"mlp","method":"risk",
             "weight":weights[scenario],"update_hours":[6,12,18],"anticipate":True,
             "raw_pv":False,"known_price":False,**kw}
        if scenario not in ("3","4-3"):job["update_hours"]=[]
        jobs.append(job)
    for scenario in SCENARIOS:
        for seed in SEEDS:add("primary",scenario,seed=seed)
        add("new_deterministic",scenario,method="deterministic",weight=0)
        add("legacy_rebased",scenario,kind="legacy",method="deterministic",weight=0)
        for kind in ("periodic","yesterday","weekly"):
            add(kind,scenario,kind=kind,method="deterministic",weight=0)
        add("risk_zero",scenario,weight=0)
        if scenario in ("3","4-3"):
            add("no_anticipation",scenario,anticipate=False)
            add("raw_forecast",scenario,raw_pv=True)
            for schedule in ([],[6],[6,12]):add("updates_"+"_".join(map(str,[0]+schedule)),scenario,update_hours=schedule)
        if scenario.startswith("4"):add("known_price",scenario,known_price=True)
    return jobs


def replay_case(arguments):
    run_id,job,stop_day=arguments;data=Data();sig=evaluation_signature(data,run_id)
    canonical={k:v for k,v in job.items() if k!="name"}
    if job["method"]=="deterministic":
        canonical.update(anticipate=False,weight=0)
    key=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()[:16]
    directory=HERE/"runs"/run_id/"evaluation"/key;directory.mkdir(parents=True,exist_ok=True)
    stamp=directory/"configuration.json"
    if stamp.exists():assert json.loads(stamp.read_text())=={"signature":sig,"case":canonical}
    else:stamp.write_text(json.dumps({"signature":sig,"case":canonical},indent=2))
    _,w=warmup(data,job["scenario"],run_id,sig);state=float(w["states"][-1,-1])
    store=LegacyStore(data) if job["kind"]=="legacy" else ForecastStore(data,run_id,job["seed"],job["kind"])
    factory=ScenarioFactory(data,store) if job["method"]=="risk" else None
    summaries=[]
    kwargs={k:job[k] for k in ("method","weight","update_hours","anticipate","raw_pv","known_price")}
    for day in range(31,stop_day):
        path=directory/f"d{day:03d}.json"
        if path.exists():
            saved=json.loads(path.read_text());summary=saved["summary"]
            assert abs(summary["initial_soc"]-state)<1e-6 and path.with_suffix(".npz").exists()
        else:
            summary,detail,logs,tree=day_run(data,store,day,job["scenario"],state,factory=factory,**kwargs)
            np.savez_compressed(path.with_suffix(".npz"),**detail)
            path.write_text(json.dumps({"summary":summary,"solvers":logs,"tree":tree},indent=2))
        summaries.append(summary);state=summary["final_soc"]
        if day in (31,58,89,119,150,180,211,242,272,303,333,364):
            print("REPLAY",job["name"],job["scenario"],job["seed"],summary["date"],flush=True)
    return {"job":job,"cache_key":key,"signature":sig,"days":len(summaries)}


def aggregate(run_id,results):
    out=ROOT/"data/results"/run_id;out.mkdir(parents=True,exist_ok=True)
    all_days=[];annual=[];solver_rows=[];trees=[]
    for result in results:
        job=result["job"];directory=HERE/"runs"/run_id/"evaluation"/result["cache_key"]
        rows=[];details=[]
        for path in sorted(directory.glob("d*.json")):
            saved=json.loads(path.read_text());row={**saved["summary"],**job};rows.append(row)
            if job["name"]=="primary":
                for log in saved["solvers"]:
                    scalar={k:v for k,v in log.items() if not isinstance(v,(dict,list))}
                    solver_rows.append({**scalar,"seed":job["seed"],"name":job["name"]})
            if job["name"]=="primary" and job["seed"]==42:
                with np.load(path.with_suffix(".npz")) as f:details.append({k:f[k].copy() for k in f.files})
                if row["date"] in ("2025-02-01","2025-03-20","2025-06-21","2025-09-23","2025-12-21"):
                    trees.append(saved["tree"])
        assert len(rows)==334
        all_days.extend(rows);frame=pd.DataFrame(rows)
        sums=("planned_kwh","final_kwh","emergency_kwh","emergency_minutes","charge_kwh","discharge_kwh",
              "surplus_kwh","planned_cost","up_cost","down_cost","emergency_cost","total_cost","violations",
              "solve_execute_seconds","solver_calls","timeout_count","fallback_count","incumbent_count","gap_certified_count")
        a={**job,**{k:float(frame[k].sum()) for k in sums},"final_soc":float(frame.iloc[-1].final_soc),
           "daily_cvar90":weighted_cvar(frame.total_cost.to_numpy(),np.full(334,1/334)),
           "worst_day_cost":float(frame.total_cost.max()),"worst_date":str(frame.loc[frame.total_cost.idxmax(),"date"]),
           "cache_key":result["cache_key"]}
        annual.append(a)
        if details:
            np.savez_compressed(out/f"dispatch_{job['scenario']}.npz",**{k:np.stack([d[k] for d in details]) for k in details[0]})
            with np.load(HERE/"runs"/run_id/"warmup"/f"{job['scenario']}.npz") as f:
                np.savez_compressed(out/f"warmup_{job['scenario']}.npz",**{k:f[k] for k in f.files})
    pd.DataFrame(all_days).to_csv(out/"daily_metrics.csv",index=False)
    pd.DataFrame(annual).to_csv(out/"dispatch_metrics.csv",index=False)
    pd.DataFrame(solver_rows).to_csv(out/"solver_metrics.csv",index=False)
    (out/"scenario_examples.json").write_text(json.dumps(trees,indent=2))
    (out/"evaluation_manifest.json").write_text(json.dumps(results,indent=2))
    forecast_metrics(run_id)


def forecast_metrics(run_id):
    from experiments.common.neural_v1.evaluate import forecast_scores
    data=Data();rows=[]
    for month in range(2,13):
        origins=month_origins(month)
        for seed in SEEDS:
            store=ForecastStore(data,run_id,seed)
            predictions=np.stack([store.get(int(o)) for o in origins])
            rows.extend(forecast_scores(data,origins,predictions,"mlp",str(seed),month))
        for kind in ("periodic","yesterday","weekly"):
            predictions=np.stack([data.baseline(int(o),kind) for o in origins])
            rows.extend(forecast_scores(data,origins,predictions,kind,"none",month))
    out=ROOT/"data/results"/run_id;frame=pd.DataFrame(rows)
    frame.to_csv(out/"forecast_metrics.csv",index=False)
    keys=["variant","seed","target","population"]
    annual=frame[frame.lead=="all"].groupby(keys).agg(n=("n","sum"),absolute_error_sum=("absolute_error_sum","sum"),
        squared_error_sum=("squared_error_sum","sum"),actual_abs_sum=("actual_abs_sum","sum")).reset_index()
    annual["mae"]=annual.absolute_error_sum/annual.n
    annual["rmse"]=np.sqrt(annual.squared_error_sum/annual.n)
    annual["wape_pct"]=100*annual.absolute_error_sum/annual.actual_abs_sum.replace(0,np.nan)
    annual.to_csv(out/"annual_forecast_metrics.csv",index=False)


def run(run_id="exp002",workers=4,stop_day=365,primary_only=False):
    calibration=json.loads((ROOT/"data/results"/run_id/"risk_calibration.json").read_text())
    weights={r["scenario"]:r["selected_weight"] for r in calibration}
    jobs=cases(weights)
    if primary_only:jobs=[j for j in jobs if j["name"]=="primary"]
    # Equivalent ablations reuse exactly the same replay rather than repeat bounded MIP searches.
    groups={}
    for job in jobs:
        canonical=json.dumps({k:v for k,v in job.items() if k!="name"},sort_keys=True)
        groups.setdefault(canonical,[]).append(job)
    results=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(replay_case,(run_id,group[0],stop_day)):group for group in groups.values()}
        for future in as_completed(pending):
            result=future.result()
            for job in pending[future]:results.append({**result,"job":job})
            print("CASE_DONE",result["job"]["name"],result["job"]["scenario"],result["job"]["seed"],flush=True)
    if stop_day==365 and not primary_only:aggregate(run_id,results)
    (HERE/"runs"/run_id/"evaluation_walltime.json").write_text(json.dumps({"seconds":time.monotonic()-began,
        "workers":workers,"case_count":len(jobs),"unique_case_count":len(groups),"complete":stop_day==365 and not primary_only},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="exp002");p.add_argument("--workers",type=int,default=4)
    p.add_argument("--stage",choices=("calibrate","evaluate","forecast"),default="evaluate")
    p.add_argument("--stop-day",type=int,default=365);p.add_argument("--primary-only",action="store_true")
    args=p.parse_args()
    if args.stage=="calibrate":calibrate(args.run_id,args.workers)
    elif args.stage=="forecast":forecast_metrics(args.run_id)
    else:run(args.run_id,args.workers,args.stop_day,args.primary_only)


if __name__=="__main__":main()
