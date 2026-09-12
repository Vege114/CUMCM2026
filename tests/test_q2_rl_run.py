"""Independent archival checks on synthetic evidence, without policy training."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.problem2.rl_planning.verify import array_hash, check_policy_audit, check_training


class RLArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.protocol = json.loads((root / "experiments/problem2/rl_planning/protocol.proposed.json").read_text())
        cls.spec = cls.protocol["runs"][0]

    def training_rows(self):
        rows = []
        for block, cutoff in enumerate(range(31, 365, 14)):
            for iteration in range(218 if block == 0 else 44):
                rows.append({"run_id": "regularized_42", "rl_seed": 42,
                             "cutoff_day": cutoff, "iteration_at_cutoff": iteration,
                             "global_iteration": len(rows), "transitions": 4608,
                             "training_start_day": max(8, cutoff-90), "training_end_day": cutoff-1,
                             "history_days": cutoff-max(8, cutoff-90), "max_observed_index": cutoff*144-1,
                             "rollout_seconds": .01, "update_seconds": .02, "mean_episode_reward": -1.,
                             "mean_episode_cost_yuan": 1000., "mean_episode_throughput_kwh": 10.,
                             "loss": 1., "policy_loss": -.1, "value_loss": 2., "entropy": 3.,
                             "approx_kl": .001, "clip_fraction": .02, "gradient_norm": .8})
        return rows

    def policy_evidence(self):
        zero = np.zeros((334, 144))
        detail = {"original": zero.copy(), "charge": zero.copy(), "discharge": zero.copy(),
                  "states": np.full((334, 145), 1200.), "price": np.ones((334, 144))}
        forecasts = np.zeros((334, 144, 2))
        audits = []
        for i, day in enumerate(range(31, 365)):
            block = (day-31)//14
            cutoff = 31+block*14
            audits.append({"day": day, "run_id": "regularized_42", "rl_seed": 42, "forecast_seed": 42,
                           "policy_training_cutoff_day": cutoff,
                           "policy_training_max_observed_index": cutoff*144-1,
                           "policy_total_iterations": 218+block*44,
                           "policy_training_origins": list(range(max(8, cutoff-90)*144, cutoff*144, 144)),
                           "midnight_purchase_sha256": array_hash(detail["original"][i]),
                           "forecast_value_sha256": array_hash(forecasts[i]),
                           "terminal_value_scope": "reward_only_not_policy_input_or_billed_credit",
                           "purchase_locked_before_actual_read": True, "actuals_in_policy_observation": False,
                           "last_evaluation_day": day == 364, "planning_initial_power_kw": 0.,
                           "planning_initial_soc": 1200.,
                           "terminal_value_yuan_per_soc_kwh": 0. if day == 364 else 1/np.sqrt(.9)})
        return detail, audits, forecasts

    def test_fixed_budget_and_future_training_detection(self):
        rows = self.training_rows()
        result = check_training(rows, self.protocol, self.spec)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["online_updates"], 23)
        self.assertEqual(result["transitions"], 5667840)
        rows[500]["max_observed_index"] += 1
        result = check_training(rows, self.protocol, self.spec)
        self.assertIn("training_max_observed_index:500", result["errors"])

    def test_nonfinite_and_duplicate_training_updates_rejected(self):
        rows = self.training_rows()
        rows[800]["loss"] = np.nan
        rows[999]["global_iteration"] = 998
        result = check_training(rows, self.protocol, self.spec)
        self.assertFalse(result["passed"])
        self.assertIn("training_nonfinite:800", result["errors"])
        self.assertIn("training_global_iteration:999", result["errors"])

    def test_midnight_hash_and_forecast_tampering_rejected(self):
        detail, audits, forecasts = self.policy_evidence()
        result = check_policy_audit(detail, audits, self.protocol, self.spec, forecasts, 0.)
        self.assertTrue(result["passed"], result["errors"])
        detail["original"][7, 83] = 1.
        forecasts[23, 19, 1] = 12.
        result = check_policy_audit(detail, audits, self.protocol, self.spec, forecasts, 0.)
        self.assertIn("policy_midnight_purchase_sha256:38", result["errors"])
        self.assertIn("policy_forecast_value_sha256:54", result["errors"])

    def test_actual_power_boundary_and_future_policy_rejected(self):
        detail, audits, forecasts = self.policy_evidence()
        audits = copy.deepcopy(audits)
        audits[10]["planning_initial_power_kw"] = 500.
        audits[40]["policy_training_cutoff_day"] = 72
        audits[-1]["terminal_value_yuan_per_soc_kwh"] = 1.
        result = check_policy_audit(detail, audits, self.protocol, self.spec, forecasts, 0.)
        self.assertIn("cross_day_actual_power_source:41", result["errors"])
        self.assertIn("policy_policy_training_cutoff_day:71", result["errors"])
        self.assertIn("terminal_reward_scope:364", result["errors"])


if __name__ == "__main__":
    unittest.main()
