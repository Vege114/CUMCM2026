import copy
import unittest

from experiments.exp008.report_payload import FinalPayloadRejected, enforce_final


class FinalizationAcceptanceTests(unittest.TestCase):
    def example(self):
        return {"missing_items": [], "manifest": {}, "scenarios": {"2": {
            "archive": {"sha256": "accepted-archive"}, "verification": {"passed": True, "goal": {"passed": False}},
            "fees": {"total_cost_yuan": 13201981.94794217},
            "battery": {"direction_reversals": 2533, "simultaneous_slots": 0}}}}

    def accept(self, payload):
        payload["manifest"]["finalization_acceptance"] = {
            "basis": "user_accepts_current_verified_result",
            "user_message": "行，我觉得现在这个结果比较满意了，就这样写报告然后提交吧",
            "accepted_q2_archive_sha256": "accepted-archive"}

    def test_old_gate_still_rejects_without_acceptance(self):
        with self.assertRaises(FinalPayloadRejected): enforce_final(self.example())

    def test_acceptance_does_not_rewrite_original_goal(self):
        p = self.example(); self.accept(p); before = copy.deepcopy(p["scenarios"])
        enforce_final(p)
        self.assertEqual(p["scenarios"], before)
        self.assertTrue(p["finalization_basis"]["user_accepted_current_verified_result"])
        self.assertFalse(p["finalization_basis"]["original_eight_percent_target_met"])

    def test_acceptance_cannot_select_another_archive(self):
        p = self.example(); self.accept(p); p["scenarios"]["2"]["archive"]["sha256"] = "different"
        with self.assertRaises(FinalPayloadRejected): enforce_final(p)

    def test_acceptance_cannot_waive_physics_or_missing_evidence(self):
        for change in ("physics", "evidence", "reversals"):
            p = self.example(); self.accept(p)
            if change == "physics": p["scenarios"]["2"]["verification"]["passed"] = False
            if change == "evidence": p["missing_items"] = [{"key": "missing", "blocks_final_payload": True}]
            if change == "reversals": p["scenarios"]["2"]["battery"]["direction_reversals"] = 3000
            with self.subTest(change=change), self.assertRaises(FinalPayloadRejected): enforce_final(p)


if __name__ == "__main__": unittest.main()
