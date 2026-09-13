from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
from experiments.exp009 import q1_cost_model as model

ROOT=Path(__file__).resolve().parents[2]


def run():
    out=ROOT/'data/results/exp009/q1'
    if out.exists():raise FileExistsError('Frozen result directory already exists')
    out.mkdir(parents=True)
    source=ROOT/'data/raw/附件1.csv'
    data=model.read_data(source);began=perf_counter()
    result=model.solve(data,delta=0.,ramp=1000.,up=3,down=2,eta_rt=.9)
    wall=perf_counter()-began
    assert len(result['stages'])==1 and result['metrics']['violations']==0
    result.update(experiment_id='exp009',objective='actual purchase cost only; no stage2',wall_seconds=wall,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__).resolve(),Path(model.__file__),ROOT/'experiments/exp009/protocol.json']},
        fixed_engineering_constraints='ramp1000/min_on3/min_off2/min_active1kW retained as exp008 assumptions')
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    (out/'cost_only.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n')
    print(json.dumps({'cost':result['metrics']['cost'],'metrics':result['metrics'],'wall_seconds':wall},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
