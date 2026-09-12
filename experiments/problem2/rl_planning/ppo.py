"""Small CPU categorical PPO policy for finite, complete daily episodes.

Import this module before creating another TensorFlow runtime if the thread limits
are required. The rollout/environment owns reward scaling and episode boundaries;
this module normalizes advantages only inside each supplied training batch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError as exc:
    raise RuntimeError("Import rl_planning.ppo before initializing TensorFlow") from exc


def generalized_advantage(rewards, values, gamma=.995, lam=.95):
    """Return GAE and value targets for complete episodes of shape ``[T, N]``.

Each column is one episode, with terminal bootstrap exactly zero. Partial
rollouts or a reset inside a column must not be passed to this function.
"""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if rewards.ndim != 2 or rewards.shape != values.shape or 0 in rewards.shape:
        raise ValueError("rewards and values must have the same nonempty [T, N] shape")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("GAE inputs must be finite")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lam must lie in [0, 1]")
    advantages = np.empty_like(rewards)
    following_value = np.zeros(rewards.shape[1], dtype=np.float32)
    following_advantage = np.zeros_like(following_value)
    for step in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[step] + gamma * following_value - values[step]
        following_advantage = delta + gamma * lam * following_advantage
        advantages[step] = following_advantage
        following_value = values[step]
    return advantages, advantages + values


class PPO:
    """Clipped categorical PPO with separate instance random streams."""

    clip_ratio = .2
    value_coefficient = .5
    entropy_coefficient = .01
    max_gradient_norm = .5
    metric_names = (
        "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
        "clip_fraction", "gradient_norm",
    )

    def __init__(self, obs_dim, action_dim, seed=42, hidden=64, learning_rate=3e-4):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        if min(self.obs_dim, self.hidden) < 1 or self.action_dim < 2:
            raise ValueError("obs_dim/hidden must be positive; action_dim must be at least two")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self._batch_rng = np.random.default_rng(self.seed)
        with tf.device("/CPU:0"):
            self._sample_rng = tf.random.Generator.from_seed(self.seed)
            inputs = tf.keras.Input(shape=(self.obs_dim,), dtype=tf.float32)
            x = inputs
            for layer in range(2):
                x = tf.keras.layers.Dense(
                    self.hidden, activation="tanh", name=f"trunk_{layer}",
                    kernel_initializer=tf.keras.initializers.Orthogonal(
                        gain=np.sqrt(2.), seed=self.seed + layer),
                    bias_initializer="zeros",
                )(x)
            logits = tf.keras.layers.Dense(
                self.action_dim, name="policy_logits",
                kernel_initializer=tf.keras.initializers.Orthogonal(
                    gain=.01, seed=self.seed + 2), bias_initializer="zeros",
            )(x)
            value = tf.keras.layers.Dense(
                1, name="value",
                kernel_initializer=tf.keras.initializers.Orthogonal(
                    gain=1., seed=self.seed + 3), bias_initializer="zeros",
            )(x)
            self.model = tf.keras.Model(inputs, (logits, value), name="categorical_ppo")
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)
            # Create Adam slots now so checkpoints can restore before any update.
            self.optimizer.build(self.model.trainable_variables)
            self._checkpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.optimizer, sample_rng=self._sample_rng)

        observations = tf.TensorSpec([None, self.obs_dim], tf.float32)
        vector = tf.TensorSpec([None], tf.float32)
        self._sample = tf.function(
            self._sample_impl, input_signature=[observations], reduce_retracing=True)
        self._greedy = tf.function(
            self._greedy_impl, input_signature=[observations], reduce_retracing=True)
        self._train = tf.function(
            self._train_impl,
            input_signature=[observations, tf.TensorSpec([None], tf.int32),
                             vector, vector, vector], reduce_retracing=True,
        )

    @staticmethod
    def _logprob(logits, actions):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)

    def _sample_impl(self, obs):
        with tf.device("/CPU:0"):
            logits, values = self.model(obs, training=False)
            # Gumbel-max categorical sampling uses the private RNG; keep logs finite.
            uniform = self._sample_rng.uniform(tf.shape(logits), minval=1e-7, maxval=1.)
            actions = tf.argmax(logits - tf.math.log(-tf.math.log(uniform)),
                                axis=-1, output_type=tf.int32)
            return actions, self._logprob(logits, actions), tf.squeeze(values, axis=-1)

    def _greedy_impl(self, obs):
        with tf.device("/CPU:0"):
            logits, values = self.model(obs, training=False)
            actions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            return actions, self._logprob(logits, actions), tf.squeeze(values, axis=-1)

    def _observations(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or not len(obs):
            raise ValueError(f"obs must have nonempty shape [N, {self.obs_dim}]")
        if not np.isfinite(obs).all():
            raise ValueError("observations must be finite")
        return obs

    def act(self, obs, deterministic=False):
        """Return action ids, their log probabilities, and state values, all [N]."""
        obs = self._observations(obs)
        output = self._greedy(obs) if deterministic else self._sample(obs)
        return tuple(item.numpy() for item in output)

    def _train_impl(self, obs, actions, old_logprob, advantages, returns):
        with tf.device("/CPU:0"), tf.GradientTape() as tape:
            logits, values = self.model(obs, training=True)
            logprob = self._logprob(logits, actions)
            log_ratio = logprob - old_logprob
            ratio = tf.exp(log_ratio)
            clipped_ratio = tf.clip_by_value(ratio, 1. - self.clip_ratio, 1. + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages,
                                                    clipped_ratio * advantages))
            value_loss = tf.reduce_mean(tf.square(tf.squeeze(values, -1) - returns))
            log_all = tf.nn.log_softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_all) * log_all, axis=-1))
            loss = (policy_loss + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy)
            loss = tf.debugging.check_numerics(loss, "nonfinite PPO objective")
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        approx_kl = tf.reduce_mean((ratio - 1.) - log_ratio)
        clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.) > self.clip_ratio, tf.float32))
        return tf.stack([loss, policy_loss, value_loss, entropy, approx_kl,
                         clip_fraction, gradient_norm])

    def update(self, batch, epochs=4, minibatch_size=512):
        """Optimize one on-policy batch; return sample-weighted minibatch metrics."""
        obs = self._observations(batch["obs"])
        count = len(obs)
        arrays = {}
        for key in ("actions", "old_logprob", "advantages", "returns"):
            raw = np.asarray(batch[key])
            if raw.shape != (count,) or not np.isfinite(raw).all():
                raise ValueError(f"{key} must be a finite vector of length {count}")
            if key == "actions":
                if not np.equal(raw, np.floor(raw)).all() or np.any(raw < 0) or np.any(raw >= self.action_dim):
                    raise ValueError("actions must be integer ids in [0, action_dim)")
                arrays[key] = raw.astype(np.int32)
            else:
                arrays[key] = raw.astype(np.float32)
        epochs, minibatch_size = int(epochs), int(minibatch_size)
        if epochs < 1 or minibatch_size < 1:
            raise ValueError("epochs and minibatch_size must be positive")
        # Only the provided training rollout contributes to this normalization.
        advantages = arrays["advantages"]
        arrays["advantages"] = ((advantages - advantages.mean(dtype=np.float64))
                                / (advantages.std(dtype=np.float64) + 1e-8)).astype(np.float32)
        total = np.zeros(len(self.metric_names), dtype=np.float64)
        for _ in range(epochs):
            order = self._batch_rng.permutation(count)
            for start in range(0, count, minibatch_size):
                ids = order[start:start + minibatch_size]
                metrics = self._train(obs[ids], arrays["actions"][ids],
                                      arrays["old_logprob"][ids], arrays["advantages"][ids],
                                      arrays["returns"][ids]).numpy()
                if not np.isfinite(metrics).all():
                    raise FloatingPointError("PPO update produced nonfinite metrics")
                total += len(ids) * metrics
        return dict(zip(self.metric_names, (total / (epochs * count)).tolist()))

    def save(self, path):
        """Save network, optimizer and both RNG states in a checkpoint directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        prefix = self._checkpoint.write(str(path / "ppo"))
        metadata = {"version": 1, "obs_dim": self.obs_dim, "action_dim": self.action_dim,
                    "hidden": self.hidden, "seed": self.seed,
                    "learning_rate": self.learning_rate,
                    "numpy_rng": self._batch_rng.bit_generator.state}
        (path / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return prefix

    def load(self, path):
        """Restore a saved directory into an instance with matching architecture."""
        path = Path(path)
        metadata = json.loads((path / "configuration.json").read_text())
        if metadata["version"] != 1 or any(
            metadata[key] != getattr(self, key) for key in ("obs_dim", "action_dim", "hidden")
        ):
            raise ValueError("checkpoint version or architecture does not match this PPO")
        self._checkpoint.read(str(path / "ppo")).assert_consumed()
        self._batch_rng.bit_generator.state = metadata["numpy_rng"]
        self.seed = metadata["seed"]
        self.learning_rate = metadata["learning_rate"]
        return self
