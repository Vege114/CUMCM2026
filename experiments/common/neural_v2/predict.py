"""Cached predictions with explicit primary policy and legacy archive adapters."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import EPOCH,HERE,ROOT,SEEDS,TARGETS,Data,month_origins,signature


class ForecastStore:
    def __init__(self,data,run_id="exp002",seed=42,kind="mlp"):
        self.data=data;self.directory=HERE/"runs"/run_id;self.seed=seed;self.kind=kind

    @lru_cache(maxsize=12)
    def month(self,month):
        path=self.directory/f"m{month:02d}_mlp_{self.seed}.npz"
        with np.load(path) as f:content={k:f[k].copy() for k in f.files}
        meta=json.loads(path.with_suffix(".json").read_text())
        assert meta["asof"]==int(month_origins(month)[0])
        assert content["signature"].item()==meta["signature"]
        content["lookup"]={int(o):i for i,o in enumerate(content["origins"])}
        return content

    def get(self,origin,issued=True,raw_pv=False,validation_month=None):
        month=int((EPOCH+pd.Timedelta(minutes=10*int(origin))).month)
        if self.kind!="mlp" or (month==1 and validation_month is None):
            return self.data.baseline(origin,kind=self.kind if self.kind!="mlp" else "periodic",issued=issued)
        pack=self.month(validation_month or month)
        p=pack["predictions"][pack["lookup"][int(origin)],:,:4 if issued else 3].copy().astype(float)
        if raw_pv and issued:
            p[:,3]=self.data.integrate_points(self.data.issued_points(origin),origin)
        return p


class LegacyStore:
    def __init__(self,data):
        self.data=data
        with np.load(ROOT/"data/results/exp001/ensemble_predictions.npz") as f:
            self.arrays={k:f[k].copy() for k in f.files}
        self.lookup={int(o):i for i,o in enumerate(self.arrays["origins"])}
        selected=pd.read_csv(ROOT/"data/results/exp001/model_selection.csv",dtype={"scenario":str})
        self.choices={(int(r.month),str(r.scenario)):r.variant for r in selected.itertuples() if r.selected}

    def get(self,origin,scenario):
        if origin<31*144:return self.data.baseline(origin,issued=scenario in ("3","4-3"))
        month=int((EPOCH+pd.Timedelta(minutes=10*int(origin))).month)
        p=self.arrays[self.choices[(month,scenario)]][self.lookup[int(origin)]].copy().astype(float)
        if scenario in ("3","4-3"):
            p[:,3]=self.data.integrate_points(p[:,3],origin)
        else:p=p[:,:3]
        return p


def run(run_id="exp002"):
    data=Data();out=ROOT/"data/results"/run_id;out.mkdir(parents=True,exist_ok=True)
    origins=np.concatenate([month_origins(m) for m in range(2,13)])
    payload={"origins":origins}
    meta=[]
    for seed in SEEDS:
        store=ForecastStore(data,run_id,seed)
        payload[f"seed_{seed}"]=np.stack([store.get(int(o)) for o in origins])
        meta.extend(json.loads((store.directory/f"m{m:02d}_mlp_{seed}.json").read_text()) for m in range(2,13))
    np.savez_compressed(out/"predictions.npz",**payload)
    (out/"training_metadata.json").write_text(json.dumps(meta,indent=2))
    (out/"data_hashes.json").write_text(json.dumps(data.hashes,ensure_ascii=False,indent=2))
    (out/"prediction_archive.json").write_text(json.dumps({"epoch":str(EPOCH),
        "targets":TARGETS,"dimensions":["origin","future_interval","target"],
        "origin_unit_minutes":10,"primary_seed":42,"averaged_seeds":False,
        "value_semantics":"load/pv historical interval power; corrected pv integrated interval average; price yuan/kWh",
        "year_end":"predictions retained beyond year end, unobserved targets excluded from scores"},indent=2))
    print("PREDICTIONS",len(origins),"origins, 3 seeds",flush=True)
