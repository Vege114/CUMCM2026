"""Audit mutation tests: failures must be visible without using the planner."""

import unittest

import numpy as np

from experiments.exp008.verify import ETA, GOAL_COST, INITIAL_SOC, goal_check, verify_arrays


def adjusted_day():
    original = np.full((1, 144), 100.0)
    final = original.copy()
    final[0, 1] = 90.0
    final[0, 2] = 110.0
    emergency = np.zeros_like(final)
    emergency[0, 3] = 5.0
    price = np.full_like(final, 2.0)
    load = 6 * (final + emergency)
    return {
        "original": original, "final": final,
        "charge": np.zeros_like(final), "discharge": np.zeros_like(final),
        "emergency": emergency, "surplus": np.zeros_like(final),
        "states": np.full((1, 145), INITIAL_SOC), "price": price,
        "actual": np.stack((load, np.zeros_like(load)), axis=-1),
        "fees": np.stack((original * price, 1.5 * price * np.maximum(final - original, 0),
                          .5 * price * np.maximum(original - final, 0), 5 * price * emergency), axis=-1),
    }


class TestExp008Verification(unittest.TestCase):
    def test_goal_uses_exp006_reversal_definition_and_separate_duration_diagnostic(self):
        battery = {"direction_reversals": 2728, "active_slots": 23027,
                   "simultaneous_slots": 0, "throughput_kwh": 11000000}
        self.assertTrue(goal_check(GOAL_COST, battery)["passed"])
        self.assertFalse(goal_check(GOAL_COST, battery, full_evaluation=False)["passed"])
        battery["active_slots"] = 23028
        self.assertTrue(goal_check(GOAL_COST, battery)["passed"])
        self.assertFalse(goal_check(GOAL_COST, battery)["stricter_all_operation_metrics_passed"])
        battery["direction_reversals"] = 2729
        self.assertFalse(goal_check(GOAL_COST, battery)["passed"])

    def test_adjustment_bill_is_compared_to_midnight_original(self):
        result = verify_arrays(adjusted_day(), "3", expected_days=1)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["billing"]["total_cost"], 28890.0)
        self.assertEqual(result["billing"]["up_cost"], 30.0)
        self.assertEqual(result["billing"]["down_cost"], 10.0)

    def test_mutated_bill_and_q2_adjustment_are_rejected(self):
        detail = adjusted_day()
        detail["fees"][0, 2, 1] -= 1
        result = verify_arrays(detail, "2", expected_days=1)
        self.assertIn("slot_settlement", result["errors"])
        self.assertIn("midnight_purchase_changed", result["errors"])
        self.assertFalse(result["goal"]["passed"])

    def test_physical_overlap_rejected_even_if_balance_and_soc_match(self):
        detail = adjusted_day()
        detail["charge"][0, 0] = 10
        detail["discharge"][0, 0] = ETA * ETA * 10
        # An extra kWh of grid supply makes the dissipative cycle balance.
        detail["actual"][0, 0, 0] -= 6 * (10 - ETA * ETA * 10)
        result = verify_arrays(detail, "3", expected_days=1)
        self.assertIn("simultaneous_charge_discharge", result["errors"])
        self.assertLess(result["max_balance_error_kwh"], 1e-6)
        self.assertLess(result["max_soc_error_kwh"], 1e-6)

    def test_future_cutoff_and_cross_day_soc_tampering_are_rejected(self):
        detail = {key: np.concatenate((value, value), axis=0) for key, value in adjusted_day().items()}
        detail["states"][1] += 1
        audit = [{"information_cutoff": 31 * 144, "max_observed_index": 31 * 144}]
        result = verify_arrays(detail, "3", expected_days=2, audit_records=audit)
        self.assertIn("future_actual_used", result["errors"])
        self.assertIn("cross_day_soc_discontinuity", result["errors"])


if __name__ == "__main__":
    unittest.main()
