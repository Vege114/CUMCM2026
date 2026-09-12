"""Frozen Q2-only monthly two-branch networks; GPU training and auditable caches."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version

from .data import HERE, ROOT, STEPS, Data, midnight_origins
from .predict import (ForecastStore, apply_residual, content_hash, file_hash,
                      historical_daylight, protocol, training_fingerprint)

os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="microgrid_q2_exp003")
class Branch(tf.keras.layers.Layer):
    def __init__(self, channel, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel

    def call(self, inputs):
        return inputs[:, :, self.channel, :]

    def get_config(self):
        return {**super().get_config(), "channel": self.channel}


@tf.keras.utils.register_keras_serializable(package="microgrid_q2_exp003")
def standardized_huber(truth, predicted):
    error = tf.abs(truth - predicted)
    return tf.reduce_mean(tf.where(error <= 1, .5 * error**2, error - .5), axis=(1, 2))


def build_model(config=None):
    cfg = protocol() if config is None else config
    architecture, training = cfg["architecture"], cfg["training"]
    if (architecture["branches"], architecture["features"], architecture["output_initialization"]) != (2, 16, "zeros"):
        raise ValueError("this implementation requires the frozen two-branch, 16-feature zero-residual model")
    if training["loss"] != "standardized_huber_delta_1":
        raise ValueError("unsupported training loss")
    inputs = tf.keras.Input((STEPS, 2, 16), name="q2_issue_time_features")
    branches = []
    for channel in range(2):
        hidden = Branch(channel)(inputs)
        for units in architecture["units"]:
            hidden = tf.keras.layers.Dense(units, activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(architecture["l2"]))(hidden)
        branches.append(tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros",
                                               name=f"residual_{channel}")(hidden))
    model = tf.keras.Model(inputs, tf.keras.layers.Concatenate()(branches))
    model.compile(optimizer=tf.keras.optimizers.Adam(training["learning_rate"]),
                  loss=standardized_huber, jit_compile=False)
    return model


def split_origins(month, config=None):
    cfg = protocol() if config is None else config
    training = cfg["training"]
    if not 2 <= month <= 12:
        raise ValueError("monthly checkpoints cover February through December")
    asof = int(midnight_origins(month)[0])
    cutoff = asof - training["validation_days"] * STEPS
    candidates = np.arange(training["earliest_training_day"] * STEPS, cutoff, dtype=int)
    hours = np.asarray(training["train_issue_hours"], dtype=int) * 6
    train = candidates[np.isin(candidates % STEPS, hours) & (candidates + STEPS <= cutoff)]
    validation = np.arange(cutoff, asof, dtype=int)
    hours = np.asarray(training["validation_issue_hours"], dtype=int) * 6
    validation = validation[np.isin(validation % STEPS, hours) & (validation + STEPS <= asof)]
    if not len(train) or not len(validation):
        raise ValueError("monthly training and validation windows must be nonempty")
    return train, validation, cutoff


def prepare_month(data, month, config=None):
    cfg = protocol() if config is None else config
    train, validation, cutoff = split_origins(month, cfg)
    formal = midnight_origins(month)
    origins = np.r_[train, validation, formal]
    features, base, mean, scale = data.features(origins, cutoff)
    truth = np.concatenate((data.labels(train, cutoff), data.labels(validation, int(formal[0]))))
    n, nv = len(train), len(validation)
    return {"x": features, "target": ((truth - base[:n + nv]) / scale).astype("float32"),
            "base": base[n:], "mean": mean, "scale": scale, "origins": origins[n:],
            "daylight_mask": historical_daylight(data, origins[n:], cfg["training"]["pv_mask_history_days"]),
            "train": train, "validation": validation, "cutoff": cutoff, "asof": int(formal[0])}


def environment(require_gpu=True):
    gpus = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpus:
        raise RuntimeError("Q2 formal training requires a TensorFlow GPU")
    tf.keras.mixed_precision.set_global_policy("float32")
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        probe = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
    if float(tf.reduce_sum(probe).numpy()) != 262144:
        raise RuntimeError("device matrix probe failed")
    packages = {}
    for name in ("tensorflow", "keras", "numpy", "scipy", "tensorflow-metal", "tensorboard"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "gpu": [str(gpu) for gpu in gpus], "matrix_device": probe.device,
            "float_policy": "float32", "jit_compile": False}


def run(run_id="exp003", months=None, seeds=None, device="gpu"):
    cfg, data = protocol(), Data()
    months = list(range(2, 13)) if months is None else months
    seeds = cfg["seed_list"] if seeds is None else seeds
    if any(seed not in cfg["seed_list"] for seed in seeds):
        raise ValueError("seeds must belong to the frozen protocol")
    env = environment(device == "gpu")
    signature, sources = training_fingerprint(data)
    directory = HERE / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps({"environment": env, "signature": signature}), flush=True)
    training = cfg["training"]
    for month in months:
        prepared = prepare_month(data, month, cfg)
        n, nv = len(prepared["train"]), len(prepared["validation"])
        for seed in seeds:
            stem = directory / f"m{month:02d}_mlp_{seed}"
            if stem.with_suffix(".json").exists():
                ForecastStore(data, seed, run_id).month(month)
                print("RESUME", stem.name, flush=True)
                continue
            if stem.with_suffix(".keras").exists() or stem.with_suffix(".npz").exists():
                raise RuntimeError(f"Incomplete checkpoint needs inspection before reuse: {stem}")
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            began = time.monotonic()
            with tf.device("/GPU:0" if device == "gpu" else "/CPU:0"):
                model = build_model(cfg)
            x, target = prepared["x"], prepared["target"]
            initial = model(x[:1], training=False).numpy()
            np.testing.assert_array_equal(initial, np.zeros_like(initial))
            before = [weight.numpy().copy() for weight in model.trainable_variables]
            dataset = tf.data.Dataset.from_tensor_slices((x[:n], target[:n]))
            dataset = dataset.shuffle(n, seed=seed).batch(training["batch_size"]).prefetch(1)
            validation = tf.data.Dataset.from_tensor_slices((x[n:n + nv], target[n:n + nv])).batch(training["batch_size"])
            fit = model.fit(dataset, validation_data=validation, epochs=training["max_epochs"], verbose=0,
                shuffle=False, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                    patience=training["patience"], restore_best_weights=True)])
            training_seconds = time.monotonic() - began
            began = time.monotonic()
            probe = model(x[n:n + 1], training=False)
            if device == "gpu" and "GPU:0" not in probe.device:
                raise RuntimeError("Q2 predictions did not execute on GPU")
            updated = any(not np.array_equal(old, weight.numpy()) for old, weight in zip(before, model.trainable_variables))
            if not updated or not np.any(probe.numpy() != initial):
                raise RuntimeError("training did not update the zero-residual model")
            raw = np.concatenate([model(x[index:index + 64], training=False).numpy()
                                  for index in range(n, len(x), 64)]) * prepared["scale"]
            raw = raw.astype("float32")
            arrays = {"origins": prepared["origins"], "base": prepared["base"], "raw_residual": raw,
                      "daylight_mask": prepared["daylight_mask"], "mean": prepared["mean"], "scale": prepared["scale"],
                      "train_cutoff": np.asarray(prepared["cutoff"]), "signature": np.asarray(signature),
                      "validation_count": np.asarray(nv)}
            arrays["predictions"] = apply_residual(arrays["base"], raw, arrays["daylight_mask"]).astype("float32")
            prediction_seconds = time.monotonic() - began
            model.save(stem.with_suffix(".keras"))
            restored = tf.keras.models.load_model(stem.with_suffix(".keras"))
            restored_probe = restored(x[n:n + 1], training=False)
            np.testing.assert_allclose(restored_probe.numpy(), probe.numpy(), atol=2e-4, rtol=2e-4)
            if device == "gpu" and "GPU:0" not in restored_probe.device:
                raise RuntimeError("reloaded Q2 checkpoint did not execute on GPU")
            np.savez_compressed(stem.with_suffix(".npz"), **arrays)
            metadata = {"signature": signature, **sources, "month": month, "seed": seed,
                "asof": prepared["asof"], "train_cutoff": prepared["cutoff"], "cutoff": prepared["cutoff"],
                "train_origins": n, "validation_origins": nv, "formal_origins": len(midnight_origins(month)),
                "train_latest_target_exclusive": int(prepared["train"][-1] + STEPS),
                "validation_latest_target_exclusive": int(prepared["validation"][-1] + STEPS),
                "train_issue_hours": training["train_issue_hours"], "validation_issue_hours": training["validation_issue_hours"],
                "epochs": len(fit.history["loss"]), "max_epochs": training["max_epochs"],
                "best_epoch": int(np.argmin(fit.history["val_loss"])) + 1, "parameters": model.count_params(),
                "history": fit.history, "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                "device": probe.device, "variable_devices": sorted({weight.value.device for weight in model.trainable_variables}),
                "zero_initial_output": True, "weights_updated": updated, "save_reload_passed": True,
                "reload_device": restored_probe.device, "environment": env,
                "weights_sha256": file_hash(stem.with_suffix(".keras")), "npz_sha256": file_hash(stem.with_suffix(".npz")),
                "prediction_content_sha256": content_hash(arrays)}
            stem.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            ForecastStore(data, seed, run_id).month(month)
            print(json.dumps({key: metadata[key] for key in ("month", "seed", "epochs", "best_epoch", "training_seconds", "parameters")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="exp003")
    parser.add_argument("--months", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    run(**vars(parser.parse_args()))
