"""Monthly, reproducible prediction-only experiment, with audited checkpoints."""

import argparse
import os
import platform
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cumcm-exp004-mpl")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from .data import (HERE, OUT, STEPS, VARIANTS, Features, age_weights, protocol,
                   sha256, source_signature, split_days, write_json)


@tf.keras.utils.register_keras_serializable(package="exp004")
class AsymmetricHuber(tf.keras.losses.Loss):
    """The 5:1 ratio is a cost-aware surrogate, not the dispatch objective."""

    def __init__(self, prices, adverse_weight=5.0, **kwargs):
        super().__init__(**kwargs)
        self.prices = list(prices)
        self.adverse_weight = float(adverse_weight)

    def call(self, truth, predicted):
        error = truth - predicted
        absolute = tf.abs(error)
        huber = tf.where(absolute <= 1, .5 * error ** 2, absolute - .5)
        adverse = tf.stack((error[:, :, 0] > 0, error[:, :, 1] < 0), axis=-1)
        weight = tf.where(adverse, self.adverse_weight, 1.)
        price = tf.constant(self.prices, dtype=predicted.dtype)
        price = price / tf.reduce_mean(price)
        return tf.reduce_mean(huber * weight * price[None, :, None], axis=(1, 2))

    def get_config(self):
        return {**super().get_config(), "prices": self.prices,
                "adverse_weight": self.adverse_weight}


def build_model(prices):
    cfg = protocol()
    arc = cfg["architecture"]
    sequence = tf.keras.Input((168, 2), name="seven_completed_days_hourly_kw")
    context = tf.keras.Input((STEPS, 2, 8), name="known_target_context")
    branches = []
    for channel, name in enumerate(("load", "pv")):
        hidden = sequence[:, :, channel:channel + 1]
        for i, dilation in enumerate(arc["dilations"]):
            hidden = tf.keras.layers.Conv1D(
                arc["filters"], arc["kernel_size"], padding="causal",
                dilation_rate=dilation, activation="relu", name=f"{name}_conv{i}",
                kernel_regularizer=tf.keras.regularizers.L2(arc["l2"]),
            )(hidden)
        hidden = tf.keras.layers.AveragePooling1D(arc["pool_size"])(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(arc["embedding_units"], activation="relu")(hidden)
        hidden = tf.keras.layers.RepeatVector(STEPS)(hidden)
        hidden = tf.keras.layers.Concatenate()([hidden, context[:, :, channel, :]])
        hidden = tf.keras.layers.Dense(arc["head_units"], activation="relu")(hidden)
        branches.append(tf.keras.layers.Dense(1, kernel_initializer="zeros",
                                              bias_initializer="zeros", name=name)(hidden))
    model = tf.keras.Model([sequence, context], tf.keras.layers.Concatenate()(branches))
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["training"]["learning_rate"]),
                  loss=AsymmetricHuber(prices), jit_compile=False)
    return model


def dataset(sequence, context, target, weight=None, *, shuffle=False, seed=42):
    values = ((sequence, context), target) if weight is None else ((sequence, context), target, weight)
    ds = tf.data.Dataset.from_tensor_slices(values)
    if shuffle:
        ds = ds.shuffle(len(sequence), seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return ds.batch(protocol()["training"]["batch_size"]).with_options(options).prefetch(1)


def run(months=None, seeds=None, variants=None):
    cfg = protocol()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(cfg["training"]["threads"])
    tf.config.threading.set_intra_op_parallelism_threads(cfg["training"]["threads"])
    tf.config.experimental.enable_op_determinism()
    features = Features()
    data = features.data
    signature, sources = source_signature(data)
    directory = HERE / "runs" / signature[:16]
    directory.mkdir(parents=True, exist_ok=True)
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "tensorflow": tf.__version__, "numpy": np.__version__, "device": "CPU",
           "threads": cfg["training"]["threads"], "deterministic_ops": True}
    write_json(directory / "environment.json", env)
    print("TRAIN_SIGNATURE", signature, flush=True)
    for month in (months or range(2, 13)):
        train, validation, formal, cutoff = split_days(month)
        days = np.r_[train, validation, formal]
        nt, nv = len(train), len(validation)
        for variant in (variants or VARIANTS):
            started = time.monotonic()
            pack = features.arrays(days, cutoff, variant)
            feature_seconds = time.monotonic() - started
            labels = data.actual[:int(formal[0]) * STEPS].reshape(-1, STEPS, 2)[days[:nt + nv]]
            target = ((labels - pack["base"][:nt + nv]) / pack["scale"]).astype("float32")
            sample_weights = age_weights(train, cutoff, cfg["training"]["history_weight_half_life_days"])
            for seed in (seeds or cfg["seed_list"]):
                stem = directory / f"{variant}_m{month:02d}_s{seed}"
                if stem.with_suffix(".json").exists():
                    import json
                    meta = json.loads(stem.with_suffix(".json").read_text())
                    if meta["signature"] != signature or any(
                        sha256(stem.with_suffix(suffix)) != meta[key]
                        for suffix, key in ((".npz", "archive_sha256"), (".keras", "weights_sha256"))
                    ):
                        raise RuntimeError(f"Changed cache: {stem}")
                    print("RESUME", stem.name, flush=True)
                    continue
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = build_model(data.fixed_price)
                x = [pack["sequence"], pack["context"]]
                initial = model([a[:1] for a in x], training=False).numpy()
                np.testing.assert_array_equal(initial, np.zeros_like(initial))
                began = time.monotonic()
                fit = model.fit(
                    dataset(x[0][:nt], x[1][:nt], target[:nt], sample_weights, shuffle=True, seed=seed),
                    validation_data=dataset(x[0][nt:nt + nv], x[1][nt:nt + nv], target[nt:]),
                    epochs=cfg["training"]["max_epochs"], verbose=0, shuffle=False,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                               patience=cfg["training"]["patience"], restore_best_weights=True)],
                )
                training_seconds = time.monotonic() - began
                began = time.monotonic()
                residual = model([a[nt + nv:] for a in x], training=False).numpy().astype(float) * pack["scale"]
                predictions = np.maximum(0, pack["base"][nt + nv:] + residual)
                predictions[:, :, 1] *= pack["mask"][nt + nv:]
                prediction_seconds = time.monotonic() - began
                if not np.isfinite(predictions).all() or not np.any(residual != 0):
                    raise RuntimeError("Training produced invalid/unchanged predictions")
                model.save(stem.with_suffix(".keras"))
                restored = tf.keras.models.load_model(stem.with_suffix(".keras"))
                np.testing.assert_allclose(
                    restored([a[nt + nv:nt + nv + 1] for a in x], training=False).numpy(),
                    model([a[nt + nv:nt + nv + 1] for a in x], training=False).numpy(), rtol=1e-6, atol=1e-6,
                )
                np.savez_compressed(stem.with_suffix(".npz"), days=formal, predictions=predictions,
                                    base=pack["base"][nt + nv:], residual=residual,
                                    seasonal_shift=pack["seasonal_shift"][nt + nv:],
                                    mask=pack["mask"][nt + nv:], mean=pack["mean"], scale=pack["scale"])
                meta = {"signature": signature, **sources, "variant": variant, "seed": seed,
                        "month": month, "asof_day": int(formal[0]), "scaler_cutoff_day": cutoff,
                        "train_days": train.tolist(), "validation_days": validation.tolist(),
                        "train_latest_label_exclusive": int((train[-1] + 1) * STEPS),
                        "validation_latest_label_exclusive": int(formal[0] * STEPS),
                        "sample_weight_min": float(sample_weights.min()),
                        "sample_weight_max": float(sample_weights.max()),
                        "uses_future_seasonal_information": variant == "oracle_season",
                        "seasonal_label_cutoff": "365 days" if variant == "oracle_season" else "strictly before each origin",
                        "parameters": model.count_params(), "epochs": len(fit.history["loss"]),
                        "best_epoch": int(np.argmin(fit.history["val_loss"])) + 1,
                        "history": fit.history, "feature_seconds_shared_across_seeds": feature_seconds,
                        "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                        "save_reload_passed": True, "environment": env,
                        "weights_sha256": sha256(stem.with_suffix(".keras")),
                        "archive_sha256": sha256(stem.with_suffix(".npz"))}
                write_json(stem.with_suffix(".json"), meta)
                print("TRAINED", variant, month, seed, meta["epochs"], round(training_seconds, 2), flush=True)
    archive_predictions(directory, signature, cfg)


def archive_predictions(directory, signature, cfg):
    import json

    metadata, archives = [], {"origins": np.arange(31, 365) * STEPS}
    for variant in VARIANTS:
        for seed in cfg["seed_list"]:
            packs = []
            for month in range(2, 13):
                stem = directory / f"{variant}_m{month:02d}_s{seed}"
                if not stem.with_suffix(".json").exists():
                    print("Partial training; formal archive not created", flush=True)
                    return
                meta = json.loads(stem.with_suffix(".json").read_text())
                if meta["signature"] != signature or sha256(stem.with_suffix(".npz")) != meta["archive_sha256"]:
                    raise RuntimeError("Invalid monthly checkpoint")
                metadata.append(meta)
                with np.load(stem.with_suffix(".npz")) as z:
                    packs.append({key: z[key].copy() for key in z.files})
            np.testing.assert_array_equal(np.concatenate([p["days"] for p in packs]), np.arange(31, 365))
            archives[f"{variant}_seed_{seed}"] = np.concatenate([p["predictions"] for p in packs])
            if seed == cfg["primary_seed"]:
                for field in ("base", "residual", "seasonal_shift", "mask"):
                    archives[f"{variant}_{field}"] = np.concatenate([p[field] for p in packs])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predictions.npz", **archives)
    write_json(OUT / "training_metadata.json", metadata)
    write_json(OUT / "prediction_manifest.json", {
        "complete": True, "signature": signature, "months": 11, "training_groups": len(metadata),
        "variants": list(VARIANTS), "seeds": cfg["seed_list"], "days": 334,
        "shape": [334, 144, 2], "dtype": "float64", "units": "kW",
        "issue": "00:00; origin is start of interval; actual indices strictly below origin",
        "archive_sha256": sha256(OUT / "predictions.npz"),
        "metadata_sha256": sha256(OUT / "training_metadata.json"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+")
    run(**vars(parser.parse_args()))
