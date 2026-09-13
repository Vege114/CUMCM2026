"""Contract checks, using historical archives only as explicit test fixtures."""
import unittest

import numpy as np

from experiments.exp008.report_payload import (BASELINE_COST, FinalPayloadRejected,
    annual_case, build_payload, component_fees, daily_tables, emergency_events,
    enforce_final, q1_case, workbook_data)


class ReportPayloadTests(unittest.TestCase):
    def test_intervals_and_units(self):
        a = {k: np.zeros((1, 144)) for k in
             ("original", "final", "charge", "discharge", "emergency")}
        a["states"] = np.full((1, 145), 6000.)
        a["fees"] = np.zeros((1, 144, 4))
        a["original"][0, 60] = 123.
        a["final"][0, 60] = 100.
        a["charge"][0, :24] = 2.
        a["fees"][0, 60] = [1., 2., 3., 4.]
        tables = daily_tables(a, 0, "test")
        self.assertEqual(tables["table1"][0], {"slot": 60, "interval": "10:00-10:10",
            "original_purchase_kwh": 123., "final_purchase_kwh": 100., "adjustment_delta_kwh": -23.})
        self.assertEqual(tables["table2"]["four_hour_blocks"][0]["charge_kwh"], 48.)
        wb = workbook_data(a, ["2025-02-01"], "3")
        self.assertEqual(wb["interval_headers"][0], "00:00-00:10")
        self.assertEqual(wb["interval_headers"][-1], "23:50-24:00")
        self.assertEqual(wb["final"][0][60], 100.)
        self.assertEqual(wb["fees"], [10.])
        self.assertEqual(component_fees(a["fees"])["adjustment_cost_yuan"], 5.)
        self.assertEqual(wb["emergency"], [["2025-02-01", "无", 0.]])

    def test_emergency_endpoint_and_gap(self):
        e = np.zeros(144)
        e[[0, 1, 143]] = [1., 2., 3.]
        events = emergency_events(e)
        self.assertEqual([r["interval"] for r in events], ["00:00-00:20", "23:50-24:00"])
        self.assertEqual([r["energy_kwh"] for r in events], [3., 3.])

    def test_unselected_preparation_preserves_six_experiments(self):
        payload = build_payload({})
        self.assertEqual(payload["scenarios"], {})
        self.assertEqual([r["experiment_id"] for r in payload["history"]["experiments"]],
                         [f"exp{i:03d}" for i in range(1, 7)])
        self.assertTrue(all(r["available"] for r in payload["history"]["experiments"]))
        with self.assertRaises(FinalPayloadRejected):
            enforce_final(payload)

    def test_existing_q1_final_second_stage_units(self):
        result = q1_case({"archive": "data/results/exp008/q1/revised.json",
                          "role": "test fixture only"}, [])
        self.assertEqual(result["final_trajectory_stage"], 2)
        self.assertTrue(result["verification"]["passed"])
        q = result["trajectory"]
        np.testing.assert_allclose(np.array(q["charge_kwh"]) * 6, q["charge_kw"])
        self.assertAlmostEqual(result["verification"]["recomputed_total_cost"],
                               result["stages"][1]["cost"])

    def test_frozen_exp006_is_not_accepted_as_improvement(self):
        case = annual_case("2", {"archive": "data/results/exp006/primary/dispatch_2.npz",
            "initial_soc_kwh": 1421.7991105135516, "role": "historical test fixture"}, [])
        self.assertTrue(case["verification"]["passed"])
        self.assertAlmostEqual(case["fees"]["total_cost_yuan"], BASELINE_COST, places=5)
        self.assertEqual(case["battery"]["direction_reversals"], 2729)
        with self.assertRaises(FinalPayloadRejected):
            enforce_final({"scenarios": {"2": case}, "missing_items": []})

    def test_exact_threshold_is_not_replaced_by_claimed_goal(self):
        fake = {"verification": {"goal": {"passed": True}},
                "fees": {"total_cost_yuan": .92 * BASELINE_COST + 1.}}
        with self.assertRaises(FinalPayloadRejected):
            enforce_final({"scenarios": {"2": fake}, "missing_items": []})


if __name__ == "__main__":
    unittest.main()
