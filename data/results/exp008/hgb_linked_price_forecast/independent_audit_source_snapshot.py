"""Independent raw-row and augmented least-squares audit of linked price fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.hgb_linked_price_forecast import LinkedPriceForecasts, OUT, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = OUT/'independent_audit.json'
    if destination.exists():
        raise FileExistsError('Independent audit is immutable')
    archive = np.load(OUT/'price_predictions.npz',allow_pickle=False)
    audited = json.loads((OUT/'prediction_audit.json').read_text())
    frozen = {p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    forecast = LinkedPriceForecasts()
    original_price = pd.read_csv(ROOT/'data/raw/附件4.csv').iloc[:,1:].to_numpy(float)
    np.testing.assert_array_equal(archive['actual'], original_price[31:])
    issued = {}
    for day in range(1,365):
        if day >= 31:
            issued[day] = forecast.store.values[forecast.store.lookup[day*144]].copy()
        else:
            values = forecast.data.reference[:,[1,2]].copy()
            # Independently reconstruct labelled January weekly-load / daily-PV.
            load_day = day-7 if day>=7 else day-1
            values[:,0] = forecast.data.actual[load_day*144:(load_day+1)*144,0]
            values[:,1] = forecast.data.actual[(day-1)*144:day*144,1]
            issued[day] = values
        np.testing.assert_array_equal(issued[day],forecast._midnight(day))
    def design(indices):
        day,slot = indices//144,indices%144
        previous=indices-144
        weekly=np.where(indices>=7*144,indices-7*144,previous)
        assert np.all(previous < day*144) and np.all(weekly < day*144)
        phase=2*np.pi*slot/144
        return np.column_stack((np.ones(len(indices)),forecast.data.fixed_price[slot],
            original_price.ravel()[previous],original_price.ravel()[weekly],
            np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),
            np.asarray([issued[int(d)][int(t),0]/1000 for d,t in zip(day,slot)]),
            np.asarray([issued[int(d)][int(t),1]/1000 for d,t in zip(day,slot)])))
    maximum_feature_error=maximum_coefficient_error=maximum_prediction_error=0.
    penalty=np.array([1e-6,5.,5.,5.,5.,5.,5.,5.,5.,5.])
    prior=np.array([0.,0.,.5,.5,0.,0.,0.,0.,0.,0.])
    for day in range(31,365):
        origin=day*144
        idx=np.arange(max(144,origin-28*144),origin)
        x=design(idx); xt=design(np.arange(origin,origin+144))
        maximum_feature_error=max(maximum_feature_error,float(np.max(np.abs(x-forecast.design(idx,origin)))))
        y=original_price.ravel()[idx]
        sw=np.sqrt(2**(-(origin-1-idx)/(14*144)))
        # Solve an augmented rectangular system via SVD least squares, not the
        # production normal-equation solve. The ridge prior is an extra label.
        xa=np.vstack((x*sw[:,None],np.diag(np.sqrt(penalty))))
        ya=np.r_[y*sw,np.sqrt(penalty)*prior]
        beta=np.linalg.lstsq(xa,ya,rcond=None)[0]
        expected=np.asarray(audited[day-31]['price_coefficients'])
        maximum_coefficient_error=max(maximum_coefficient_error,float(np.max(np.abs(beta-expected))))
        p=np.maximum(1e-4,np.einsum('ni,i->n',xt,beta,optimize=False))
        maximum_prediction_error=max(maximum_prediction_error,float(np.max(np.abs(p-archive['linked_hgb'][day-31]))))
    assert maximum_feature_error==0.
    assert maximum_coefficient_error<1e-8 and maximum_prediction_error<1e-8
    effects=[]
    for day in (31,90,151,243,364):
        original=forecast.get(day,scenario='4-2')['price']
        changed=LinkedPriceForecasts()
        current=changed.store.lookup[day*144]
        changed.store.values[current,:,0]+=1000.
        different=changed.get(day,scenario='4-2')['price']
        beta=float(audited[day-31]['price_coefficients'][-2])
        np.testing.assert_allclose(different,np.maximum(1e-4,original+beta),rtol=0,atol=1e-10)
        assert np.max(np.abs(different-original))>1e-6
        effects.append({'day':day,'current_known_load_forecast_mutation_kw':1000,
            'maximum_price_response_yuan_per_kwh':float(np.max(np.abs(different-original))),
            'expected_coefficient_response_yuan_per_kwh':beta})
    assert all(sha(OUT/name)==h for name,h in frozen.items())
    result={'passed':True,'raw_price_csv_sha256':sha(ROOT/'data/raw/附件4.csv'),
        'existing_artifacts_unchanged_sha256':frozen,
        'all364_own_issued_load_PV_features_reconstructed':True,
        'all334_designs_equal_independent_raw_row_build':True,
        'maximum_feature_error':maximum_feature_error,
        'maximum_augmented_lstsq_coefficient_error':maximum_coefficient_error,
        'maximum_augmented_lstsq_prediction_error':maximum_prediction_error,
        'known_current_forecast_feature_dependence_nonvacuous_checks':effects,
        'source_sha256':sha(__file__)}
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'independent_audit_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('passed','maximum_feature_error',
        'maximum_augmented_lstsq_coefficient_error','maximum_augmented_lstsq_prediction_error')},indent=2))


if __name__=='__main__':
    main()
