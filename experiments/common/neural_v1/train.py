"""GPU training with immutable month/variant/seed checkpoints and causal validation."""

import argparse
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from .data import ROOT, SOURCE_CHANNELS, Data, month_origins, split_origins

os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

VARIANTS = ("mlp", "gru", "tcn", "gru_no_calendar", "gru_one_day")
SEEDS = (42, 2026, 3407)


@tf.keras.utils.register_keras_serializable(package="microgrid")
class Columns(tf.keras.layers.Layer):
    def __init__(self, start, stop, **kwargs):
        super().__init__(**kwargs)
        self.start, self.stop = start, stop

    def call(self, inputs):
        return inputs[..., self.start:self.stop]

    def get_config(self):
        return {**super().get_config(), "start": self.start, "stop": self.stop}


def build_model(variant, hours):
    history = tf.keras.Input((hours, 3), name="completed_hourly_history")
    known = tf.keras.Input((144, 17), name="issue_time_known_features")
    calendar = Columns(0, 5)(known)
    outputs = []
    for j, channel in enumerate(SOURCE_CHANNELS):
        h = Columns(channel, channel + 1)(history)
        if variant == "mlp":
            h = tf.keras.layers.Flatten()(h)
            h = tf.keras.layers.Dense(64, activation="relu")(h)
            h = tf.keras.layers.Dense(32, activation="relu")(h)
        elif variant == "tcn":
            for rate in (1, 2, 4, 8):
                h = tf.keras.layers.Conv1D(32, 3, padding="causal", dilation_rate=rate,
                                           activation="relu")(h)
            h = tf.keras.layers.GlobalAveragePooling1D()(h)
        else:
            h = tf.keras.layers.GRU(32)(h)
        h = tf.keras.layers.RepeatVector(144)(h)
        lag = Columns(5 + j * 3, 8 + j * 3)(known)
        h = tf.keras.layers.Concatenate()([h, calendar, lag])
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        outputs.append(tf.keras.layers.Dense(1, name=f"residual_{j}")(h))
    model = tf.keras.Model([history, known], tf.keras.layers.Concatenate()(outputs))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", jit_compile=False)
    return model


def predict(model, h, k, b, scale):
    chunks = []
    for i in range(0, len(h), 64):
        chunks.append(model([h[i:i + 64], k[i:i + 64]], training=False).numpy())
    raw = np.concatenate(chunks) * scale[np.array(SOURCE_CHANNELS)] + b
    return np.maximum(raw, 0).astype("float32")


def environment(require_gpu=True):
    gpus = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpus:
        raise RuntimeError("GPU required. NVIDIA: use scripts/setup_ml.sh tf in WSL2/Linux; "
                           "Apple Silicon: check tensorflow-metal.")
    if tf.test.is_built_with_cuda():
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("float32")
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        probe = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
    assert float(tf.reduce_sum(probe).numpy()) == 262144
    packages = ["tensorflow", "keras", "numpy", "scipy", "tensorboard"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        packages.append("tensorflow-metal")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in packages},
            "tensorflow_build": tf.sysconfig.get_build_info(),
            "gpu": [str(g) for g in gpus], "matrix_device": probe.device,
            "float_policy": "float32", "jit_compile": False}


def run(args):
    env = environment(args.device == "gpu")
    print(json.dumps({"environment": env}, ensure_ascii=False), flush=True)
    data = Data()
    run_dir = ROOT / "experiments/common/neural_v1/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        b"".join((Path(__file__).parent / n).read_bytes() for n in ("train.py", "data.py"))
        + json.dumps(data.hashes, sort_keys=True).encode()
    ).hexdigest()
    (run_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2))
    for month in args.months:
        train, val, cutoff = split_origins(month)
        test = month_origins(month)
        for variant in args.variants:
            history_days = 1 if variant == "gru_one_day" else 7
            opts = {"history_days": history_days, "calendar": variant != "gru_no_calendar"}
            origins = np.r_[train, val, test]
            h, k, b, mean, scale = data.features(origins, cutoff, **opts)
            n, nv = len(train), len(val)
            target = (data.labels(np.r_[train, val]) - b[:n + nv]) / scale[np.array(SOURCE_CHANNELS)]
            target = target.astype("float32")
            for seed in args.seeds:
                destination = run_dir / f"m{month:02d}_{variant}_{seed}"
                metadata_path = destination.with_suffix(".json")
                if metadata_path.exists():
                    meta = json.loads(metadata_path.read_text())
                    if meta["signature"] != signature or meta["max_epochs"] != args.epochs:
                        raise RuntimeError(f"Stale checkpoint {destination}; choose a new run-id.")
                    print(f"RESUME {destination.name}", flush=True)
                    continue
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                began = time.monotonic()
                with tf.device("/GPU:0" if args.device == "gpu" else "/CPU:0"):
                    model = build_model(variant, history_days * 24)
                dataset = tf.data.Dataset.from_tensor_slices(((h[:n], k[:n]), target[:n]))
                dataset = dataset.shuffle(n, seed=seed).batch(64).prefetch(1)
                validation = tf.data.Dataset.from_tensor_slices(
                    ((h[n:n + nv], k[n:n + nv]), target[n:n + nv])).batch(64).prefetch(1)
                before = model.trainable_variables[-1].numpy().copy()
                fit = model.fit(dataset, validation_data=validation, epochs=args.epochs,
                                callbacks=[tf.keras.callbacks.EarlyStopping(
                                    monitor="val_loss", patience=6, restore_best_weights=True)],
                                verbose=0, shuffle=False)
                elapsed = time.monotonic() - began
                predictions = predict(model, h[n:], k[n:], b[n:], scale)
                out = model([h[n:n + 1], k[n:n + 1]], training=False)
                if args.device == "gpu" and "GPU:0" not in out.device:
                    raise RuntimeError("Prediction did not execute on GPU")
                assert np.isfinite(predictions).all()
                assert not np.array_equal(before, model.trainable_variables[-1].numpy())
                model.save(destination.with_suffix(".keras"))
                restored = tf.keras.models.load_model(destination.with_suffix(".keras"))
                np.testing.assert_allclose(restored([h[n:n + 1], k[n:n + 1]]).numpy(),
                                           out.numpy(), rtol=2e-4, atol=2e-4)
                np.savez_compressed(destination.with_suffix(".npz"),
                                    origins=np.r_[val, test], predictions=predictions,
                                    validation_count=nv, mean=mean, scale=scale)
                meta = {"signature": signature, "month": month, "variant": variant,
                        "seed": seed, "train_origins": n, "validation_origins": nv,
                        "test_origins": len(test), "train_cutoff": cutoff,
                        "asof": int(test[0]), "max_epochs": args.epochs,
                        "epochs": len(fit.history["loss"]),
                        "best_epoch": int(np.argmin(fit.history["val_loss"])) + 1,
                        "seconds": elapsed, "parameters": model.count_params(),
                        "history": fit.history, "device": out.device,
                        "variable_devices": sorted({v.value.device for v in model.trainable_variables}),
                        "save_reload_passed": True, "environment": env}
                metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                print(json.dumps({key: meta[key] for key in
                                  ("month", "variant", "seed", "epochs", "best_epoch", "seconds")}),
                      flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="exp001")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--months", nargs="+", type=int, default=list(range(2, 13)))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=60)
    run(p.parse_args())


if __name__ == "__main__":
    main()
