"""Numerical checks on tiny synthetic batches, never a formal training run."""

import tempfile
import unittest

import numpy as np

from experiments.problem2.rl_planning.ppo import PPO, generalized_advantage


class PPOTests(unittest.TestCase):
    def make_batch(self, policy, count=32):
        rng = np.random.default_rng(17)
        obs = rng.normal(size=(count, policy.obs_dim)).astype(np.float32)
        actions, logprob, values = policy.act(obs)
        advantages = rng.normal(size=count).astype(np.float32)
        return {"obs": obs, "actions": actions, "old_logprob": logprob,
                "advantages": advantages, "returns": values + advantages}

    def test_gae_zero_terminal_and_discounted_returns(self):
        rewards = np.array([[1., 4.], [2., 5.], [3., 6.]], dtype=np.float32)
        values = np.array([[.2, 3.], [.4, 2.], [.9, 1.]], dtype=np.float32)
        advantages, returns = generalized_advantage(rewards, values, gamma=.9, lam=1.)
        expected = np.array([[1 + .9 * 2 + .81 * 3, 4 + .9 * 5 + .81 * 6],
                             [2 + .9 * 3, 5 + .9 * 6], [3, 6]])
        np.testing.assert_allclose(returns, expected, rtol=1e-6)
        np.testing.assert_allclose(advantages[-1], rewards[-1] - values[-1])
        delta, _ = generalized_advantage(rewards, values, gamma=.9, lam=0.)
        np.testing.assert_allclose(delta[:-1], rewards[:-1] + .9 * values[1:] - values[:-1])

    def test_seed_repeat_and_on_policy_ratio(self):
        first, second = PPO(7, 5, seed=9), PPO(7, 5, seed=9)
        obs = np.random.default_rng(2).normal(size=(24, 7)).astype(np.float32)
        for _ in range(2):
            initial, repeated = first.act(obs), second.act(obs)
            for left, right in zip(initial, repeated):
                np.testing.assert_array_equal(left, right)
        actions, old_logprob, _ = initial
        logits = first.model(obs, training=False)[0].numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        recomputed = np.log(probabilities[np.arange(len(obs)), actions])
        np.testing.assert_allclose(np.exp(recomputed - old_logprob), 1., atol=2e-7)
        np.testing.assert_array_equal(first.act(obs, deterministic=True)[0], logits.argmax(axis=1))
        first.act(obs[:3])
        self.assertEqual(first._sample.experimental_get_tracing_count(), 1)

    def test_update_changes_parameters_and_has_finite_metrics(self):
        policy = PPO(7, 5, seed=5)
        batch = self.make_batch(policy)
        before = [weight.numpy().copy() for weight in policy.model.trainable_variables]
        metrics = policy.update(batch, epochs=2, minibatch_size=11)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertGreater(metrics["entropy"], 0.)
        self.assertGreaterEqual(metrics["approx_kl"], -1e-7)
        self.assertTrue(0 <= metrics["clip_fraction"] <= 1)
        self.assertTrue(any(not np.array_equal(old, new.numpy())
                            for old, new in zip(before, policy.model.trainable_variables)))
        self.assertEqual(int(policy.optimizer.iterations.numpy()), 6)
        self.assertEqual(policy._train.experimental_get_tracing_count(), 1)

    def test_checkpoint_preserves_optimizer_and_random_streams(self):
        original = PPO(7, 5, seed=11)
        batch = self.make_batch(original)
        original.update(batch, epochs=1, minibatch_size=16)
        with tempfile.TemporaryDirectory(prefix="q2-ppo-test-") as directory:
            original.save(directory)
            restored = PPO(7, 5, seed=99).load(directory)
            self.assertEqual(int(original.optimizer.iterations.numpy()),
                             int(restored.optimizer.iterations.numpy()))
            for left, right in zip(original.act(batch["obs"]), restored.act(batch["obs"])):
                np.testing.assert_array_equal(left, right)
            first = original.update(batch, epochs=1, minibatch_size=13)
            second = restored.update(batch, epochs=1, minibatch_size=13)
            np.testing.assert_allclose(list(first.values()), list(second.values()), rtol=1e-6)
            for left, right in zip(original.model.get_weights(), restored.model.get_weights()):
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
