"""Run from the repository root: uv run --locked python scripts/check_environment.py."""

import argparse
import importlib
import os
import platform
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly allow a CPU-only check")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    os.environ.setdefault("KERAS_HOME", str(root / ".cache" / "keras"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("CUDA_CACHE_PATH", str(root / ".cache" / "cuda"))

    print(
        f"Python {platform.python_version()} / {platform.system()} {platform.machine()}", flush=True
    )
    for package, module in (
        ("tensorflow", "tensorflow"),
        ("tensorboard", "tensorboard"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("openpyxl", "openpyxl"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
    ):
        importlib.import_module(module)
        print(f"  {package} {version(package)}", flush=True)

    import gymnasium as gym
    import matplotlib
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not args.allow_cpu and not gpus:
        raise RuntimeError(
            "GPU required. NVIDIA: use WSL2/Linux and scripts/setup_ml.sh tf; "
            "Apple Silicon: check tensorflow-metal. Use --allow-cpu only intentionally."
        )
    if tf.test.is_built_with_cuda():
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow build: {tf.sysconfig.get_build_info()}", flush=True)
    tf.keras.mixed_precision.set_global_policy("float32")
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        probe = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
    if gpus and "GPU:0" not in probe.device:
        raise RuntimeError("GPU matrix operation was not placed on GPU")
    print(f"PASS device placement: {probe.device}", flush=True)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    y = x.sum(axis=1, keepdims=True)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", jit_compile=False)
    before = [weight.numpy().copy() for weight in model.trainable_weights]
    loss = float(model.train_on_batch(x, y))
    if not np.isfinite(loss) or not any(
        np.any(old != new.numpy()) for old, new in zip(before, model.trainable_weights)
    ):
        raise RuntimeError("TensorFlow training failed: non-finite loss or unchanged weights")

    with TemporaryDirectory(prefix="cumcm-environment-") as temp:
        model_path = Path(temp) / "smoke.keras"
        model.save(model_path)
        restored = tf.keras.models.load_model(model_path)
        np.testing.assert_allclose(restored(x).numpy(), model(x).numpy(), rtol=1e-5, atol=1e-6)
        writer = tf.summary.create_file_writer(str(Path(temp) / "tensorboard"))
        with writer.as_default():
            tf.summary.scalar("smoke/loss", loss, step=0)
        writer.close()
        if not list((Path(temp) / "tensorboard").glob("events.out.tfevents.*")):
            raise RuntimeError("TensorBoard did not create an event file")
    print(f"PASS TensorFlow training, model save/load, TensorBoard (loss={loss:.4f})", flush=True)

    # GRU and convolution exercise cuDNN, which a matrix multiply alone does not test.
    sequence_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(16, 3)),
            tf.keras.layers.Conv1D(8, 3, activation="relu"),
            tf.keras.layers.GRU(8),
            tf.keras.layers.Dense(1),
        ]
    )
    sequence_model.compile(optimizer="adam", loss="mse", jit_compile=False)
    sequence_loss = float(sequence_model.train_on_batch(
        rng.normal(size=(4, 16, 3)).astype(np.float32), np.ones((4, 1), dtype=np.float32)
    ))
    if not np.isfinite(sequence_loss):
        raise RuntimeError("Conv1D/GRU training returned non-finite loss")
    print(f"PASS Conv1D + GRU forward/backward (loss={sequence_loss:.4f})", flush=True)

    # Exercise the TensorFlow -> Gymnasium interface, without training a task-specific policy.
    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=42)
        policy = tf.keras.Sequential([tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(2)])
        for _ in range(16):
            action = int(tf.argmax(policy(observation[None, :])[0]).numpy())
            observation, reward, terminated, truncated, _ = env.step(action)
            if not env.observation_space.contains(observation) or not np.isfinite(reward):
                raise RuntimeError("Gymnasium returned an invalid observation or reward")
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
    print("PASS TensorFlow policy inference + Gymnasium reset/step", flush=True)

    frame = pd.DataFrame({"power_kw": [1.0, 2.0], "energy_kwh": [0.25, 0.5]})
    buffer = BytesIO()
    frame.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    pd.testing.assert_frame_equal(
        pd.read_excel(buffer, engine="openpyxl"), frame, check_dtype=False
    )
    for number in range(1, 5):
        pd.read_excel(root / "data" / "raw" / f"附件{number}.xlsx", nrows=3, engine="openpyxl")
    print("PASS Excel read/write and original attachments 1-4", flush=True)

    fig, ax = plt.subplots()
    ax.plot(frame["power_kw"], frame["energy_kwh"])
    figure_buffer = BytesIO()
    fig.savefig(figure_buffer, format="png")
    plt.close(fig)
    if not figure_buffer.getvalue().startswith(b"\x89PNG"):
        raise RuntimeError("Matplotlib did not produce a PNG")
    print("PASS plotting", flush=True)
    print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
    print("Environment ready. Smoke checks passed; no experiment results were generated.")


if __name__ == "__main__":
    main()
