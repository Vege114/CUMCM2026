"""Independent checks from the published timing example and protocol corner cases."""
import copy
import unittest

from evaluate import evaluate


def event(path, request_id, virtual_time, position=None, channel=None, **response_fields):
    request = {"request_id": request_id, "arena_id": "default", "robot_id": "REDACTED"}
    if position is not None:
        request["position"] = dict(zip(("x", "y"), position))
        request["channel"] = channel
    response = {"accepted": True, "virtual_time_s": virtual_time, **response_fields}
    return {"path": path, "request": request, "response": response, "http_status": 200}


def metadata(**extra):
    return {"run_id": "unit", "problem": 3, "mode": "offline", "source": "local_mock",
            "total_jammers": 10, "total_source": "unit_fixture", **extra}


def sample_events():
    return [
        event("/enter", "enter", 0, real_timestamp_ms=1000, remaining_real_duration_s=1200),
        event("/measure", "m1", 105, (300, 400), 1, measure_result="no_signal"),
        event("/measure", "m2", 111, (300, 400), 2, measure_result="direction", svd_deg=123.45),
        event("/clear", "c1", 194, (300, 0), 3, clear_result="no_target_in_range"),
        event("/measure", "m3", 199, (300, 0), 2, measure_result="no_signal"),
        event("/exit", "exit", 199, real_timestamp_ms=2500, exit_reason="user_exit"),
    ]


class EvaluatorTests(unittest.TestCase):
    def test_published_199_second_example(self):
        result = evaluate(sample_events(), metadata())
        self.assertTrue(result["validation"]["valid_run"])
        self.assertEqual(result["official_metrics"]["virtual_total_time_s"], 199)
        self.assertEqual(result["official_metrics"]["program_runtime_s"], 1.5)
        self.assertEqual(result["diagnostics"]["travel_distance_m"], 900)
        self.assertEqual(result["diagnostics"]["channel_switch_count"], 1)
        self.assertEqual(result["diagnostics"]["final_measure_channel"], 2)
        self.assertEqual(result["time_breakdown_s"], {
            "movement": 180, "channel_switch": 1, "measurement": 15,
            "optical_localization": 3, "laser_clearance": 0,
        })
        self.assertIsNone(result["official_metrics"]["average_localization_clear_time_s"])

    def test_idempotent_replay_does_not_double_count(self):
        events = sample_events()
        events.insert(3, copy.deepcopy(events[2]))
        result = evaluate(events, metadata())
        self.assertTrue(result["validation"]["valid_run"])
        self.assertEqual(result["diagnostics"]["measure_count"], 3)
        self.assertEqual(result["diagnostics"]["duplicate_response_count"], 1)

    def test_rejected_zero_does_not_rewind_time(self):
        events = sample_events()
        rejected = event("/measure", "rejected", 0, (0, 0), 4)
        rejected["response"]["accepted"] = False
        events.insert(4, rejected)
        result = evaluate(events, metadata())
        self.assertTrue(result["validation"]["valid_run"])
        self.assertEqual(result["diagnostics"]["rejected_request_count"], 1)
        self.assertEqual(result["diagnostics"]["observed_virtual_time_s"], 199)

    def test_success_metric_denominator_and_post_clear_search(self):
        events = [
            event("/enter", "e", 0, remaining_real_duration_s=1200),
            event("/clear", "c", 5, (0, 0), 7, clear_result="success"),
            event("/measure", "m", 10, (0, 0), 1, measure_result="no_signal"),
            event("/exit", "x", 10, exit_reason="user_exit"),
        ]
        result = evaluate(events, metadata())
        self.assertTrue(result["validation"]["valid_run"])
        self.assertEqual(result["official_metrics"]["cleared_ratio"], 0.1)
        self.assertEqual(result["official_metrics"]["average_localization_clear_time_s"], 10)
        self.assertEqual(result["diagnostics"]["channel_switch_count"], 0)

    def test_formal_total_remains_unknown(self):
        result = evaluate(sample_events(), metadata(mode="formal", source="official_simulator"))
        self.assertIsNone(result["official_metrics"]["total_jammers"])
        self.assertIsNone(result["official_metrics"]["cleared_ratio"])
        self.assertIsNone(result["diagnostics"]["all_cleared"])

    def test_partial_run_does_not_claim_final_time(self):
        result = evaluate(sample_events()[:-1], metadata())
        self.assertFalse(result["validation"]["valid_run"])
        self.assertIsNone(result["official_metrics"]["virtual_total_time_s"])
        self.assertEqual(result["diagnostics"]["observed_virtual_time_s"], 199)

    def test_corrupted_clock_fails_validation(self):
        events = sample_events()
        events[3]["response"]["virtual_time_s"] = 191
        result = evaluate(events, metadata())
        self.assertFalse(result["validation"]["valid_run"])
        self.assertTrue(result["validation"]["errors"])


if __name__ == "__main__":
    unittest.main()
