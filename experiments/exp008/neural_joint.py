"""One fixed joint load/PV CNN with symmetric tariff and net-demand losses.

The exp004 causal no-season feature, split, age-weighting and CPU protocol is
preserved. This candidate changes representation sharing and loss together;
it is not a single-factor attribution study. Nothing is written to exp004.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.problem2.exp003.data import ROOT
from experiments.problem2.exp004.data import (
    STEPS,
    Features,
    age_weights,
    protocol,
    sha256,
    split_days,
    write_json,
)
from experiments.problem2.exp004.predict import ForecastStore

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import tensorflow as tf

OUT=ROOT/'data/results/exp008/neural_joint'
CANDIDATE={'variant':'no_season','seed':42,'shared_channels':['load','pv'],
           'filters':8,'kernel_size':5,'dilations':[1,2],'padding':'causal',
           'pool_size':6,'embedding_units':16,'head_units':16,'l2':.0001,
           'component_huber_group_weight':1.,'net_huber_group_weight':1.,
           'net_normalization':'standard deviation of observed net kW before monthly training cutoff',
           'loss':'normalized_tariff * (mean(two symmetric channel Hubers) + net Huber)',
           'joint_changes_not_single_factor_attribution':True,
           'hyperparameter_search':False,'official_pv_or_future_prices_used':False,
           'yearly_seasonal_features_or_labels_used':False,
           'evaluation_role':'2025 development, not untouched independent validation'}


@tf.keras.utils.register_keras_serializable(package='exp008')
class JointTariffHuber(tf.keras.losses.Loss):
    def __init__(self,prices,component_scales,net_scale,**kwargs):
        super().__init__(**kwargs)
        self.prices=list(prices)
        self.component_scales=list(component_scales)
        self.net_scale=float(net_scale)

    @staticmethod
    def huber(error):
        absolute=tf.abs(error)
        return tf.where(absolute<=1.,.5*error**2,absolute-.5)

    def call(self,truth,predicted):
        error=truth-predicted
        component=tf.reduce_mean(self.huber(error),axis=-1)
        scales=tf.constant(self.component_scales,dtype=predicted.dtype)
        # Targets and outputs are residual z-scores; their difference converts
        # to physical kW with channel scales. The periodic base cancels out.
        net_error=(error[:,:,0]*scales[0]-error[:,:,1]*scales[1])/self.net_scale
        prices=tf.constant(self.prices,dtype=predicted.dtype)
        weights=prices/tf.reduce_mean(prices)
        return tf.reduce_mean(weights[None,:]*(component+self.huber(net_error)),axis=1)

    def get_config(self):
        return {**super().get_config(),'prices':self.prices,
                'component_scales':self.component_scales,'net_scale':self.net_scale}


def build_model(prices,component_scales,net_scale):
    sequence=tf.keras.Input((168,2),name='seven_completed_days_hourly_kw')
    context=tf.keras.Input((STEPS,2,8),name='known_target_context')
    hidden=sequence
    for i,dilation in enumerate(CANDIDATE['dilations']):
        hidden=tf.keras.layers.Conv1D(8,5,padding='causal',dilation_rate=dilation,
            activation='relu',name=f'joint_conv{i}',kernel_regularizer=tf.keras.regularizers.L2(.0001))(hidden)
    hidden=tf.keras.layers.AveragePooling1D(6,name='joint_pool')(hidden)
    hidden=tf.keras.layers.Flatten(name='joint_flatten')(hidden)
    hidden=tf.keras.layers.Dense(16,activation='relu',name='joint_embedding')(hidden)
    hidden=tf.keras.layers.RepeatVector(STEPS,name='joint_repeated_embedding')(hidden)
    branches=[]
    for channel,name in enumerate(('load','pv')):
        branch=tf.keras.layers.Concatenate(name=f'{name}_embedding_context')([hidden,context[:,:,channel,:]])
        branch=tf.keras.layers.Dense(16,activation='relu',name=f'{name}_head')(branch)
        branches.append(tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name=name)(branch))
    model=tf.keras.Model([sequence,context],tf.keras.layers.Concatenate(name='joint_outputs')(branches))
    model.compile(optimizer=tf.keras.optimizers.Adam(protocol()['training']['learning_rate']),
                  loss=JointTariffHuber(prices,component_scales,net_scale),jit_compile=False)
    return model


def dataset(sequence,context,target,weight=None,*,shuffle=False):
    values=((sequence,context),target) if weight is None else ((sequence,context),target,weight)
    ds=tf.data.Dataset.from_tensor_slices(values)
    if shuffle:
        ds=ds.shuffle(len(sequence),seed=42)
    options=tf.data.Options();options.threading.private_threadpool_size=1
    return ds.batch(protocol()['training']['batch_size']).with_options(options).prefetch(1)


def configure_cpu():
    cfg=protocol()['training']
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(cfg['threads'])
    tf.config.threading.set_intra_op_parallelism_threads(cfg['threads'])
    tf.config.experimental.enable_op_determinism()
    return {'python':platform.python_version(),'platform':platform.platform(),
            'tensorflow':tf.__version__,'numpy':np.__version__,'device':'CPU',
            'threads':cfg['threads'],'deterministic_ops':True,
            'visible_gpus':len(tf.config.get_visible_devices('GPU')),
            'onednn_options':os.environ['TF_ENABLE_ONEDNN_OPTS']}


def loss_check():
    prices=np.array([.4,1.2]);scales=np.array([100.,300.]);net_scale=250.
    y=np.array([[[1.,-1.],[2.,.5]]],dtype='float32')
    loss=JointTariffHuber(prices,scales,net_scale)
    measured=float(loss.call(tf.constant(y),tf.zeros_like(y)).numpy()[0])
    symmetric=float(loss.call(tf.constant(-y),tf.zeros_like(y)).numpy()[0])
    def huber(error):
        return np.where(np.abs(error)<=1,.5*error**2,np.abs(error)-.5)
    net=(y[:,:,0]*100-y[:,:,1]*300)/250
    expected=float(np.mean(prices/prices.mean()*(huber(y).mean(-1)+huber(net))))
    assert abs(measured-expected)<1e-6 and measured==symmetric
    return {'passed':True,'physical_unit_and_price_manual_loss_error':abs(measured-expected),
            'sign_symmetry_error':abs(measured-symmetric),'expected_loss':expected}


def feature_pollution_check(features,day,cutoff,model):
    original=features.arrays(np.array([day]),cutoff,'no_season')
    altered=copy.copy(features.data)
    altered.actual=features.data.actual.copy()
    altered.actual[day*STEPS:]+=50000.
    changed=Features(altered).arrays(np.array([day]),cutoff,'no_season')
    errors={key:float(np.max(np.abs(original[key].astype(float)-changed[key].astype(float))))
            for key in ('sequence','context','base','seasonal_shift','mask','mean','scale')}
    if any(value!=0. for value in errors.values()):
        raise AssertionError(errors)
    a=model([original['sequence'],original['context']],training=False).numpy()
    b=model([changed['sequence'],changed['context']],training=False).numpy()
    difference=float(np.max(np.abs(a-b)))
    assert difference==0.
    return {'passed':True,'mutated_actual_from_exclusive_issue':day*STEPS,
            'feature_errors':errors,'current_model_output_error':difference}


def signature(data):
    paths=[Path(__file__),ROOT/'experiments/problem2/exp004/data.py',
           ROOT/'experiments/problem2/exp004/protocol.json',ROOT/'experiments/problem2/exp003/data.py']
    source={str(path.relative_to(ROOT)):sha256(path) for path in paths}
    payload={'candidate':CANDIDATE,'inherited_training':protocol()['training'],
             'source_sha256':source,'data_sha256':data.hashes}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload


def train_month(month,features,directory,run_signature):
    stem=directory/f'joint_m{month:02d}_s42'
    if stem.with_suffix('.json').exists():
        meta=json.loads(stem.with_suffix('.json').read_text())
        assert meta['signature']==run_signature
        assert sha256(stem.with_suffix('.npz'))==meta['archive_sha256']
        assert sha256(stem.with_suffix('.keras'))==meta['weights_sha256']
        print('RESUME',month,flush=True)
        return meta
    cfg=protocol()['training'];data=features.data
    train,validation,formal,cutoff=split_days(month)
    days=np.r_[train,validation,formal];nt,nv=len(train),len(validation)
    pack=features.arrays(days,cutoff,'no_season')
    labels=data.actual[:int(formal[0])*STEPS].reshape(-1,STEPS,2)[days[:nt+nv]]
    target=((labels-pack['base'][:nt+nv])/pack['scale']).astype('float32')
    observed_training=data.actual[:cutoff*STEPS]
    net_scale=max(1.,float((observed_training[:,0]-observed_training[:,1]).std()))
    weights=age_weights(train,cutoff,cfg['history_weight_half_life_days'])
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
    model=build_model(data.fixed_price,pack['scale'],net_scale)
    x=[pack['sequence'],pack['context']]
    initial=model([a[:1] for a in x],training=False).numpy()
    assert np.array_equal(initial,np.zeros_like(initial))
    began=perf_counter()
    fit=model.fit(dataset(x[0][:nt],x[1][:nt],target[:nt],weights,shuffle=True),
                  validation_data=dataset(x[0][nt:nt+nv],x[1][nt:nt+nv],target[nt:]),
                  epochs=cfg['max_epochs'],verbose=0,shuffle=False,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                    patience=cfg['patience'],restore_best_weights=True)])
    elapsed=perf_counter()-began
    residual=model([a[nt+nv:] for a in x],training=False).numpy().astype(float)*pack['scale']
    predictions=np.maximum(0.,pack['base'][nt+nv:]+residual)
    predictions[:,:,1]*=pack['mask'][nt+nv:]
    assert np.isfinite(predictions).all() and np.any(residual!=0.)
    model.save(stem.with_suffix('.keras'))
    restored=tf.keras.models.load_model(stem.with_suffix('.keras'))
    before=model([a[nt+nv:] for a in x],training=False).numpy()
    after=restored([a[nt+nv:] for a in x],training=False).numpy()
    reload_error=float(np.max(np.abs(before-after)))
    assert reload_error<=1e-6
    pollution=feature_pollution_check(features,int(formal[0]),cutoff,restored)
    np.savez_compressed(stem.with_suffix('.npz'),days=formal,predictions=predictions,
        base=pack['base'][nt+nv:],residual=residual,mask=pack['mask'][nt+nv:],
        mean=pack['mean'],scale=pack['scale'],net_scale=np.array(net_scale))
    metadata={'signature':run_signature,'month':month,'variant':'no_season','seed':42,
        'asof_day':int(formal[0]),'scaler_cutoff_day':cutoff,
        'train_days':train.tolist(),'validation_days':validation.tolist(),
        'train_latest_label_exclusive':int((train[-1]+1)*STEPS),
        'validation_latest_label_exclusive':int(formal[0]*STEPS),
        'component_scales_kw':pack['scale'].tolist(),'net_scale_kw':net_scale,
        'net_scaler_latest_label_exclusive':cutoff*STEPS,
        'history_weight_half_life_days':90,'sample_weight_min':float(weights.min()),
        'sample_weight_max':float(weights.max()),'parameters':model.count_params(),
        'epochs':len(fit.history['loss']),'best_epoch':int(np.argmin(fit.history['val_loss']))+1,
        'history':fit.history,'training_seconds':elapsed,
        'save_reload_passed':True,'save_reload_max_error':reload_error,
        'future_feature_pollution_check':pollution,
        'weights_sha256':sha256(stem.with_suffix('.keras')),
        'archive_sha256':sha256(stem.with_suffix('.npz')),
        'uses_future_seasonal_information':False}
    write_json(stem.with_suffix('.json'),metadata)
    print('TRAINED',month,'epochs',metadata['epochs'],'seconds',round(elapsed,2),
          'reload',reload_error,'future',pollution['current_model_output_error'],flush=True)
    return metadata


def score(values,truth,price):
    error=values-truth;net=error[:,:,0]-error[:,:,1]
    high=price>=np.quantile(price,.75)
    energy=net.sum(axis=1)/6;prefix=np.cumsum(net/6,axis=1)
    return {'net_rmse_kw':float(np.sqrt(np.mean(net**2))),
            'price_weighted_net_rmse_kw':float(np.sqrt(np.mean(price*net**2)/price.mean())),
            'load_rmse_kw':float(np.sqrt(np.mean(error[:,:,0]**2))),
            'pv_rmse_kw':float(np.sqrt(np.mean(error[:,:,1]**2))),
            'net_bias_kw':float(net.mean()),'high_price_threshold':float(np.quantile(price,.75)),
            'high_price_net_rmse_kw':float(np.sqrt(np.mean(net[:,high]**2))),
            'high_price_net_mae_kw':float(np.abs(net[:,high]).mean()),
            'high_price_net_bias_kw':float(net[:,high].mean()),
            'daily_net_energy_rmse_kwh':float(np.sqrt(np.mean(energy**2))),
            'daily_net_energy_mae_kwh':float(np.abs(energy).mean()),
            'intraday_cumulative_error_rmse_kwh':float(np.sqrt(np.mean(prefix**2))),
            'cumulative_net_energy_error_kwh':float(energy.sum())}


def archive_and_score(features,directory,run_signature):
    metadata=[];predictions=[];days=[]
    for month in range(2,13):
        stem=directory/f'joint_m{month:02d}_s42'
        if not stem.with_suffix('.json').exists():
            continue
        m=json.loads(stem.with_suffix('.json').read_text())
        assert m['signature']==run_signature and sha256(stem.with_suffix('.npz'))==m['archive_sha256']
        with np.load(stem.with_suffix('.npz')) as archive:
            predictions.append(archive['predictions']);days.extend(archive['days'].tolist())
        metadata.append(m)
    all_days=np.asarray(days);values=np.concatenate(predictions)
    complete=len(all_days)==334 and np.array_equal(all_days,np.arange(31,365))
    filename='predictions.npz' if complete else 'partial_predictions.npz'
    np.savez_compressed(OUT/filename,origins=all_days*STEPS,values=values)
    # Targets enter this scoring stage only after every archived output for its
    # issue has been computed without that issue's future truth.
    truth=features.data.actual[all_days[:,None]*STEPS+np.arange(STEPS)]
    baseline=ForecastStore('no_season',seed=42).values[all_days-31]
    ridge=CalibratedStore('ridge_28').values[all_days-31]
    annual=[];monthly=[];daily=[]
    for name,predicted in [('cnn_raw',baseline),('ridge28',ridge),('joint_cnn',values)]:
        annual.append({'name':name,'days':len(days),**score(predicted,truth,features.data.fixed_price)})
        dates=pd.Timestamp('2025-01-01')+pd.to_timedelta(all_days,unit='D')
        for month in sorted(set(dates.month)):
            ids=dates.month==month
            monthly.append({'name':name,'month':month,'days':int(ids.sum()),
                            **score(predicted[ids],truth[ids],features.data.fixed_price)})
        for i,date in enumerate(dates):
            daily.append({'name':name,'date':str(date.date()),
                          **score(predicted[i:i+1],truth[i:i+1],features.data.fixed_price)})
    prefix='' if complete else 'partial_'
    pd.DataFrame(annual).to_csv(OUT/f'{prefix}annual_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/f'{prefix}monthly_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/f'{prefix}daily_metrics.csv',index=False)
    result={'complete':complete,'months':len(metadata),'days':len(days),'signature':run_signature,
            'run_directory':str(directory),'archive_sha256':sha256(OUT/filename),
            'all_cpu_save_reload_and_causality_checks_passed':all(m['save_reload_passed'] and
                    m['future_feature_pollution_check']['passed'] for m in metadata),
            'net_rmse_below_ridge28':annual[-1]['net_rmse_kw']<annual[1]['net_rmse_kw'],
            'price_weighted_net_rmse_below_ridge28':annual[-1]['price_weighted_net_rmse_kw']<annual[1]['price_weighted_net_rmse_kw'],
            'metrics':annual,'cnn_retrained':True,'original_exp004_overwritten':False,
            'single_factor_attribution_claimed':False,'independent_validation_claimed':False}
    write_json(OUT/f'{prefix}manifest.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def run(months=None):
    environment=configure_cpu();features=Features()
    run_signature,sources=signature(features.data)
    directory=OUT/'runs'/run_signature[:16];directory.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',{'candidate':CANDIDATE,'inherited_training':protocol()['training'],
        'inherited_train_validation_split':'monthly; last7 complete days validation; scales stop before validation',
        'loss_override':'symmetric component group plus equal-weight net group; original asymmetric5 is not used',
        'signature':run_signature,**sources})
    write_json(directory/'environment.json',environment)
    write_json(directory/'loss_check.json',loss_check())
    (directory/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    preserved=ROOT/'data/results/exp004/predictions.npz';original_hash=sha256(preserved)
    for month in (months or range(2,13)):
        train_month(month,features,directory,run_signature)
    if sha256(preserved)!=original_hash:
        raise AssertionError('Original exp004 prediction archive changed')
    write_json(directory/'original_archive_unchanged.json',{'passed':True,'sha256':original_hash})
    return archive_and_score(features,directory,run_signature)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--months',type=int,nargs='+')
    args=parser.parse_args();run(args.months)
