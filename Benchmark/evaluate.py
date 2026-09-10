"""Evaluate one public-protocol run without simulator internals or dependencies."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
ACTION_PATHS = {"/enter", "/measure", "/clear", "/exit"}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(numerator: float, denominator: float | None) -> float | None:
    return numerator / denominator if denominator is not None and denominator > 0 else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            events.append(event)
    return events


@dataclass
class ReplayState:
    """Only accepted, unique actions can alter physical state."""

    position: tuple[float, float] = (0.0, 0.0)
    channel: int = 1
    distance_m: float = 0.0
    counts: Counter = field(default_factory=Counter)
    measure_results: Counter = field(default_factory=Counter)
    cleared_channels: set[int] = field(default_factory=set)
    seen: dict[str, str] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=lambda: {
        "movement": 0.0, "channel_switch": 0.0, "measurement": 0.0,
        "optical_localization": 0.0, "laser_clearance": 0.0,
    })
    entered: bool = False
    exited: bool = False
    enter_timestamp_ms: float | None = None
    exit_timestamp_ms: float | None = None
    virtual_time_s: float | None = None
    remaining_real_duration_s: float | None = None
    exit_reason: str | None = None
    timeline: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def reconstructed_time_s(self) -> float:
        return sum(self.breakdown.values())


def move_for_action(state: ReplayState, request: dict[str, Any], index: int) -> bool:
    position = request.get("position")
    if not isinstance(position, dict) or not all(
        finite_number(position.get(axis)) and abs(position[axis]) <= 2_000_000
        for axis in ("x", "y")
    ):
        state.errors.append(f"event {index}: accepted action has invalid position")
        return False
    target = float(position["x"]), float(position["y"])
    distance = math.dist(state.position, target)
    state.distance_m += distance
    state.breakdown["movement"] += distance / 5.0
    state.position = target
    return True


def apply_action(state: ReplayState, path: str, request: dict[str, Any],
                 response: dict[str, Any], index: int) -> None:
    if path == "/enter":
        if state.entered:
            state.errors.append(f"event {index}: more than one accepted enter in one run")
        state.entered = True
        timestamp = response.get("real_timestamp_ms")
        state.enter_timestamp_ms = timestamp if finite_number(timestamp) else None
        budget = response.get("remaining_real_duration_s")
        if finite_number(budget) and 0 <= budget <= 1200:
            state.remaining_real_duration_s = budget
        else:
            state.warnings.append("enter response has no valid remaining_real_duration_s")
    elif path == "/exit":
        if not state.entered:
            state.errors.append(f"event {index}: accepted exit without enter")
        state.exited = True
        timestamp = response.get("real_timestamp_ms")
        state.exit_timestamp_ms = timestamp if finite_number(timestamp) else None
        state.exit_reason = response.get("exit_reason")
        if state.exit_reason != "user_exit":
            state.errors.append(f"event {index}: unexpected successful exit reason")
    else:
        if not state.entered or state.exited:
            state.errors.append(f"event {index}: action outside entered session")
        channel = request.get("channel")
        if not finite_number(channel) or channel != int(channel) or not 1 <= channel <= 20:
            state.errors.append(f"event {index}: accepted action has invalid channel")
            return
        channel = int(channel)
        if not move_for_action(state, request, index):
            return
        if path == "/measure":
            state.counts["measure_count"] += 1
            if channel != state.channel:
                state.counts["channel_switch_count"] += 1
                state.breakdown["channel_switch"] += 1
            state.channel = channel
            state.breakdown["measurement"] += 5
            result = response.get("measure_result")
            if result not in {"direction", "near", "no_signal"}:
                state.errors.append(f"event {index}: invalid measure_result")
            else:
                state.measure_results[result] += 1
            if result == "direction":
                bearing = response.get("svd_deg")
                if not finite_number(bearing) or not 0 <= bearing < 360:
                    state.errors.append(f"event {index}: invalid direction bearing")
        else:
            state.counts["clear_attempt_count"] += 1
            state.breakdown["optical_localization"] += 3
            result = response.get("clear_result")
            if result == "success":
                state.counts["clear_success_count"] += 1
                state.breakdown["laser_clearance"] += 2
                if channel in state.cleared_channels:
                    state.errors.append(f"event {index}: a channel was cleared twice with different IDs")
                state.cleared_channels.add(channel)
                state.timeline.append({
                    "virtual_time_s": response.get("virtual_time_s"),
                    "cleared_count": len(state.cleared_channels),
                    "channel": channel,
                })
            elif result == "no_target_in_range":
                state.counts["clear_failure_count"] += 1
            else:
                state.errors.append(f"event {index}: invalid clear_result")


def replay(events: Iterable[dict[str, Any]]) -> ReplayState:
    state = ReplayState()
    for index, event in enumerate(events, 1):
        state.counts["request_attempt_count"] += 1
        path, request, response = event.get("path"), event.get("request"), event.get("response")
        status = event.get("http_status", 200)
        if response is None:
            state.counts["transport_error_count"] += 1
            continue
        if not isinstance(response, dict) or not isinstance(request, dict):
            state.errors.append(f"event {index}: request/response must be JSON objects")
            continue
        if status != 200 or response.get("accepted") is not True:
            state.counts["rejected_request_count"] += 1
            continue
        if path not in ACTION_PATHS:
            state.errors.append(f"event {index}: unknown accepted action path")
            continue
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            state.errors.append(f"event {index}: accepted action lacks request_id")
            continue
        signature = json.dumps([path, request], sort_keys=True, ensure_ascii=False)
        if request_id in state.seen:
            if signature != state.seen[request_id]:
                state.errors.append(f"event {index}: accepted request_id reused with changed content")
            state.counts["duplicate_response_count"] += 1
            continue
        state.seen[request_id] = signature
        state.counts["accepted_unique_action_count"] += 1
        apply_action(state, path, request, response, index)
        virtual_time = response.get("virtual_time_s")
        if not finite_number(virtual_time) or virtual_time < 0:
            state.errors.append(f"event {index}: invalid accepted virtual_time_s")
            continue
        if state.virtual_time_s is not None and virtual_time < state.virtual_time_s - 0.000001:
            state.errors.append(f"event {index}: accepted virtual clock moved backwards")
        state.virtual_time_s = float(virtual_time)
        tolerance = 0.0001 + len(state.seen) * 0.000001
        if abs(state.virtual_time_s - state.reconstructed_time_s) > tolerance:
            state.errors.append(f"event {index}: virtual clock does not match physical action accounting")
    return state


def resolve_total(metadata: dict[str, Any], state: ReplayState) -> int | None:
    total = metadata.get("total_jammers")
    if metadata.get("mode") == "formal":
        if total is not None:
            state.warnings.append("formal truth is not disclosed; supplied total_jammers was ignored")
        return None
    if total is None:
        state.warnings.append("source total is unknown; clearance ratio and all_cleared are unavailable")
        return None
    if not finite_number(total) or total != int(total) or not 10 <= total <= 16:
        state.errors.append("total_jammers must be an integer from 10 to 16")
        return None
    if not metadata.get("total_source"):
        state.warnings.append("total_jammers lacks total_source provenance")
    return int(total)


def resolve_runtime(metadata: dict[str, Any], state: ReplayState) -> tuple[float | None, str | None]:
    supplied = metadata.get("program_runtime_s")
    if finite_number(supplied) and supplied >= 0:
        source = metadata.get("program_runtime_source", "metadata_claimed")
        if source != "simulator_ui":
            state.warnings.append("program runtime supplied without simulator_ui provenance")
        return float(supplied), source
    if state.enter_timestamp_ms is not None and state.exit_timestamp_ms is not None:
        delta = (state.exit_timestamp_ms - state.enter_timestamp_ms) / 1000.0
        if delta < 0:
            state.errors.append("server timestamps imply negative runtime")
            return None, None
        return delta, "server_timestamp_proxy"
    return None, None


def evaluate(events: Iterable[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-compatible measured metrics; unknown values remain None."""
    state = replay(events)
    total = resolve_total(metadata, state)
    cleared = len(state.cleared_channels)
    if total is not None and cleared > total:
        state.errors.append("observed cleared count exceeds provided total")
    ui_cleared = metadata.get("official_cleared_count")
    if ui_cleared is not None and ui_cleared != cleared:
        state.errors.append("UI cleared count disagrees with accepted event log")
    final_virtual = state.virtual_time_s if state.exited else None
    supplied_virtual = metadata.get("final_virtual_time_s")
    if supplied_virtual is not None:
        if finite_number(supplied_virtual) and supplied_virtual >= 0:
            if state.exited and abs(supplied_virtual - (final_virtual or 0)) > 0.01:
                state.errors.append("UI final virtual time disagrees with exit response")
            elif not state.exited:
                final_virtual = float(supplied_virtual)
                state.warnings.append("final time comes from metadata; incomplete exit log limits replay validation")
        else:
            state.errors.append("invalid final_virtual_time_s metadata")
    if not state.entered:
        state.errors.append("accepted enter is missing")
    if not state.exited:
        state.warnings.append("accepted exit is missing; observed action time may only be a lower bound")
    if state.counts["transport_error_count"]:
        state.warnings.append("transport errors occurred; verify retry recovery and UI outcome")
    runtime, runtime_source = resolve_runtime(metadata, state)
    if metadata.get("problem") not in (3, 4):
        state.errors.append("metadata.problem must be 3 or 4")
    if metadata.get("mode") not in {"practice", "formal", "offline"}:
        state.errors.append("metadata.mode must be practice, formal, or offline")
    if metadata.get("source") not in {"official_simulator", "local_mock"}:
        state.errors.append("metadata.source must explicitly identify official_simulator or local_mock")
    observed = state.virtual_time_s
    balance_error = observed - state.reconstructed_time_s if observed is not None else None
    diagnostics = {key: state.counts[key] for key in (
        "request_attempt_count", "accepted_unique_action_count", "duplicate_response_count",
        "rejected_request_count", "transport_error_count", "measure_count", "channel_switch_count",
        "clear_attempt_count", "clear_success_count", "clear_failure_count",
    )}
    diagnostics.update({
        "travel_distance_m": state.distance_m,
        "measure_results": {key: state.measure_results[key] for key in ("direction", "near", "no_signal")},
        "no_signal_ratio": ratio(state.measure_results["no_signal"], state.counts["measure_count"]),
        "all_cleared": cleared == total if total is not None else None,
        "cleared_channels": sorted(state.cleared_channels),
        "observed_virtual_time_s": observed,
        "reconstructed_virtual_time_s": state.reconstructed_time_s,
        "time_balance_error_s": balance_error,
        "remaining_real_duration_s_at_enter": state.remaining_real_duration_s,
        "final_position": {"x": state.position[0], "y": state.position[1]},
        "final_measure_channel": state.channel,
        "end_reason": state.exit_reason or metadata.get("end_reason"),
    })
    complete = state.entered and state.exited and not state.errors
    if total is not None and cleared < total:
        state.warnings.append("not all sources were cleared")
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "official_metrics": {
            "case_code": metadata.get("case_code"), "cleared_count": cleared,
            "total_jammers": total, "cleared_ratio": ratio(cleared, total),
            "virtual_total_time_s": final_virtual,
            "average_localization_clear_time_s": ratio(final_virtual, cleared) if final_virtual is not None else None,
            "program_runtime_s": runtime, "program_runtime_source": runtime_source,
        },
        "diagnostics": diagnostics,
        "time_breakdown_s": state.breakdown,
        "clearance_timeline": state.timeline,
        "validation": {
            "valid_run": complete, "complete_event_log": complete,
            "all_cleared_verified": diagnostics["all_cleared"] is True and complete,
            "formal_submission_ready": False,
            "formal_submission_note": "正式交付仍需逐局核对 UI 表格与原名加密日志，不由该评价器自动认证。",
            "errors": list(dict.fromkeys(state.errors)), "warnings": list(dict.fromkeys(state.warnings)),
        },
        "comparison": {
            "stratum": {key: metadata.get(key) for key in ("problem", "source", "mode")},
            "paired_case_key": metadata.get("seed") if metadata.get("source") == "local_mock" else None,
            "priority": ["all_cleared", "cleared_ratio", "average_localization_clear_time_s"],
            "official_score": None,
            "note": "全清优先是内部比较规则；官方不同随机案例不可声称配对改进。",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    result = evaluate(read_jsonl(args.events), metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "valid_run": result["validation"]["valid_run"],
                      **result["official_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
