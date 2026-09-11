"""Perturb all unseen data and verify every month's actual model inputs and labels."""

import argparse
import copy
import json

import numpy as np

from .data import ROOT, SOURCE_CHANNELS, Data, month_origins, split_origins


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='exp001');args=p.parse_args()
    data=Data();rows=[]
    for month in range(2,13):
        asof=int(month_origins(month)[0]);train,val,cutoff=split_origins(month)
        changed=copy.deepcopy(data)
        changed.actual[asof:]=changed.actual[asof:]*7+123
        for origin in changed.forecasts:
            if origin>asof:changed.forecasts[origin]=changed.forecasts[origin]*11+321
        origins=np.r_[train,val,asof]
        for history_days,calendar in ((7,True),(7,False),(1,True)):
            before=data.features(origins,cutoff,history_days,calendar)
            after=changed.features(origins,cutoff,history_days,calendar)
            for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
            y=(data.labels(np.r_[train,val])-before[2][:-1])/before[4][np.array(SOURCE_CHANNELS)]
            z=(changed.labels(np.r_[train,val])-after[2][:-1])/after[4][np.array(SOURCE_CHANNELS)]
            np.testing.assert_array_equal(y,z)
        rows.append({'month':month,'decision_origin':asof,'train_cutoff':int(cutoff),
                     'train_origins':len(train),'validation_origins':len(val),
                     'perturbation':'All actual values from decision onward, all forecasts published later',
                     'unchanged':['training inputs','completed training labels','validation inputs and labels',
                                  'normalization statistics','month-start inference inputs'],
                     'feature_configurations':3,'status':'passed'})
        print(f'CAUSAL INPUTS month={month}: passed',flush=True)
    out=ROOT/'data/results'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    (out/'causality_checks.json').write_text(json.dumps({'months':rows,'status':'passed',
        'interpretation':'Identical inputs, labels and random seeds define the same learning problem; GPU numerical nondeterminism is not asserted absent.',
        'selection_and_plan':'Separate unit tests perturb future data and verify validation selection and prior plans.'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
