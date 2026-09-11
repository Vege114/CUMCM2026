"""Exactly one fixed four-branch MLP per month and seed; GPU is mandatory by default."""
import argparse
import json
import os
import time

from .data import CHANNELS, HERE, ROOT, SEEDS, Data, month_origins, signature, split_origins

os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="microgrid_v2")
class Branch(tf.keras.layers.Layer):
    def __init__(self, channel, **kwargs):
        super().__init__(**kwargs); self.channel = channel
    def call(self, inputs):
        return inputs[:, :, self.channel, :]
    def get_config(self):
        return {**super().get_config(), "channel": self.channel}


@tf.keras.utils.register_keras_serializable(package="microgrid_v2")
def endpoint_huber(truth, predicted):
    error = tf.abs(truth - predicted)
    loss = tf.where(error <= 1, 0.5 * error**2, error - 0.5)
    weights = np.ones((144,4), "float32")
    weights[:, 3] = 0; weights[5::6,3] = 6
    return tf.reduce_mean(loss * tf.constant(weights), axis=(1,2))


def build_model():
    inputs = tf.keras.Input((144, 4, 16), name="issue_time_features")
    branches = []
    for channel in range(4):
        h = Branch(channel)(inputs)
        for units in (32, 16):
            h = tf.keras.layers.Dense(units, activation="relu",
                                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(h)
        branches.append(tf.keras.layers.Dense(1, kernel_initializer="zeros",
                                               bias_initializer="zeros", name=f"residual_{channel}")(h))
    model = tf.keras.Model(inputs, tf.keras.layers.Concatenate()(branches))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=endpoint_huber, jit_compile=False)
    return model


def run(run_id="exp002", months=range(2,13), seeds=SEEDS, epochs=60, device="gpu"):
    from experiments.common.neural_v1.train import environment
    env = environment(device == "gpu")
    data = Data(); directory = HERE / "runs" / run_id; directory.mkdir(parents=True, exist_ok=True)
    sig = signature(("train.py", "data.py", "protocol.json"), {"data":data.hashes, "epochs":epochs})
    (directory / "environment.json").write_text(json.dumps(env, indent=2))
    for month in months:
        train, val, cutoff = split_origins(month)
        # Include inference-only validation tail without using incomplete labels for early stopping.
        inference = np.arange(cutoff, month_origins(month)[-1]+1, 36)
        origins = np.r_[train, inference]
        x, base, mean, scale = data.features(origins, cutoff)
        n, nv = len(train), len(val)
        truth = (data.labels(np.r_[train,val])-base[:n+nv]) / scale[CHANNELS]
        for seed in seeds:
            stem = directory / f"m{month:02d}_mlp_{seed}"
            if stem.with_suffix(".json").exists():
                meta = json.loads(stem.with_suffix(".json").read_text())
                if meta["signature"] != sig:
                    raise RuntimeError("Checkpoint signature differs; use a new run-id")
                assert stem.with_suffix(".keras").exists() and stem.with_suffix(".npz").exists()
                print("RESUME", stem.name, flush=True); continue
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
            began = time.monotonic()
            with tf.device("/GPU:0" if device == "gpu" else "/CPU:0"):
                model = build_model()
            initial = model(x[:1], training=False).numpy()
            np.testing.assert_array_equal(initial, np.zeros_like(initial))
            training = tf.data.Dataset.from_tensor_slices((x[:n], truth[:n].astype("float32")))
            training = training.shuffle(n, seed=seed).batch(64).prefetch(1)
            validation = tf.data.Dataset.from_tensor_slices((x[n:n+nv], truth[n:n+nv].astype("float32"))).batch(64)
            fit = model.fit(training, validation_data=validation, epochs=epochs, verbose=0, shuffle=False,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
            training_seconds = time.monotonic()-began
            began = time.monotonic()
            output = model(x[n:n+1], training=False)
            if device == "gpu" and "GPU:0" not in output.device:
                raise RuntimeError("GPU output required")
            assert np.any(output.numpy() != initial)
            raw = np.concatenate([model(x[i:i+64], training=False).numpy()
                                  for i in range(n,len(x),64)]) * scale[CHANNELS] + base[n:]
            raw = np.maximum(raw,0)
            predictions = raw.copy()
            for i,o in enumerate(inference):
                predictions[i,:,3] = data.interpolate_hourly(raw[i,5::6,3], int(o))
                ids = int(o)+np.arange(144)
                history_days = min(28, int(o)//144)
                history_ids = ids[None,:] - np.arange(1,history_days+1)[:,None]*144
                allowed = data.actual[np.maximum(history_ids,0),1].max(axis=0)>0
                predictions[i,~allowed,1]=0
                # Hourly correction is constrained at night using only past daylight and the issue.
                hourly = raw[i,5::6,3].copy()
                hourly[~(allowed[5::6] | (data.forecasts[int(o)]>0))]=0
                predictions[i,:,3] = data.interpolate_hourly(hourly,int(o))
                raw[i,5::6,3] = hourly
            assert np.isfinite(predictions).all()
            prediction_seconds = time.monotonic()-began
            model.save(stem.with_suffix(".keras"))
            restored = tf.keras.models.load_model(stem.with_suffix(".keras"))
            np.testing.assert_allclose(restored(x[n:n+1]).numpy(), output.numpy(), atol=2e-4, rtol=2e-4)
            np.savez_compressed(stem.with_suffix(".npz"), origins=inference, predictions=predictions.astype("float32"),
                hourly_pv=raw[:,5::6,3].astype("float32"), baseline=base[n:], mean=mean, scale=scale,
                signature=sig, train_cutoff=cutoff, validation_count=nv)
            meta = {"signature":sig, "month":month, "seed":seed, "asof":int(month_origins(month)[0]),
                    "train_cutoff":cutoff, "train_origins":n, "validation_origins":nv,
                    "epochs":len(fit.history["loss"]), "best_epoch":int(np.argmin(fit.history["val_loss"]))+1,
                    "training_seconds":training_seconds, "prediction_seconds":prediction_seconds,
                    "parameters":model.count_params(), "history":fit.history,
                    "device":output.device, "zero_initial_output":True,
                    "weights_updated":True, "save_reload_passed":True, "environment":env}
            stem.with_suffix(".json").write_text(json.dumps(meta,indent=2))
            print(json.dumps({k:meta[k] for k in ("month","seed","epochs","training_seconds","parameters")}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="exp002")
    p.add_argument("--months",nargs="+",type=int,default=list(range(2,13)))
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--device",choices=("gpu","cpu"),default="gpu")
    args=p.parse_args();run(**vars(args))


if __name__=="__main__":main()
