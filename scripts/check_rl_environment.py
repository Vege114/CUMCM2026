"""Verify real SB3 DQN training, CUDA/cuDNN, checkpoint reload and environment interaction."""

import argparse
import os
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(root / ".cache" / "cuda"))

    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env

    print(f"Python {platform.python_version()} / {platform.system()}", flush=True)
    for package in ("stable-baselines3", "gymnasium", "torch", "tensorboard", "numpy"):
        print(f"  {package} {version(package)}", flush=True)
    available = torch.cuda.is_available()
    if not available and not args.allow_cpu:
        raise RuntimeError("CUDA required. Run scripts/setup_ml.sh rl and check nvidia-smi.")
    device = torch.device("cuda:0" if available else "cpu")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    probe = torch.ones((64, 64), device=device) @ torch.ones((64, 64), device=device)
    if probe.sum().item() != 262144 or probe.device.type != device.type:
        raise RuntimeError("Matrix operation failed or used the wrong device")
    convolution = torch.nn.Conv2d(3, 8, 3).to(device)
    conv_loss = convolution(torch.randn((4, 3, 16, 16), device=device)).square().mean()
    conv_loss.backward()
    if convolution.weight.grad is None or not torch.isfinite(convolution.weight.grad).all():
        raise RuntimeError("Convolution backward failed")
    if available:
        torch.cuda.synchronize()
        print(f"GPU: {torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda} / "
              f"cuDNN {torch.backends.cudnn.version()}", flush=True)
    print(f"PASS matrix + convolution forward/backward on {device}", flush=True)

    env = gym.make("CartPole-v1")
    try:
        check_env(env.unwrapped, warn=True)
        with TemporaryDirectory(prefix="cumcm-rl-") as temp:
            model = DQN(
                "MlpPolicy", env, device=device, seed=42, learning_starts=32,
                buffer_size=512, batch_size=32, train_freq=4, gradient_steps=1,
                policy_kwargs={"net_arch": [32, 32]}, tensorboard_log=temp, verbose=0,
            )
            before = [weight.detach().clone() for weight in model.q_net.parameters()]
            model.learn(total_timesteps=256, log_interval=1)
            if model.num_timesteps < 256 or model._n_updates == 0:
                raise RuntimeError("DQN did not perform training updates")
            if not any(not torch.equal(old, new) for old, new in
                       zip(before, model.q_net.parameters())):
                raise RuntimeError("DQN weights did not change")
            if not all(torch.isfinite(weight).all() for weight in model.q_net.parameters()):
                raise RuntimeError("DQN produced non-finite weights")
            if next(model.q_net.parameters()).device.type != device.type:
                raise RuntimeError("DQN policy is on the wrong device")
            observation, _ = env.reset(seed=42)
            action, _ = model.predict(observation, deterministic=True)
            path = Path(temp) / "dqn.zip"
            model.save(path)
            restored = DQN.load(path, device=device)
            restored_action, _ = restored.predict(observation, deterministic=True)
            np.testing.assert_array_equal(action, restored_action)
            for key, value in model.q_net.state_dict().items():
                torch.testing.assert_close(value, restored.q_net.state_dict()[key])
            observation, reward, _, _, _ = env.step(int(restored_action))
            if not env.observation_space.contains(observation) or not np.isfinite(reward):
                raise RuntimeError("Restored policy failed environment interaction")
            if not list(Path(temp).rglob("events.out.tfevents.*")):
                raise RuntimeError("TensorBoard event file missing")
            print(f"PASS DQN: {model.num_timesteps} steps, {model._n_updates} updates, "
                  f"policy on {next(model.q_net.parameters()).device}", flush=True)
            print("PASS model save/load, TensorBoard, Gymnasium interaction", flush=True)
    finally:
        env.close()
    print("RL environment ready. This smoke run does not measure task performance.")


if __name__ == "__main__":
    main()
