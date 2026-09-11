"""Schema boundary around v1 HTTP transport and exact idempotent retries."""
import math

from experiments.baseline_v1.baseline.protocol import Client, HttpTransport, ProtocolError, TransportUnavailable


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CheckedTransport:
    def __init__(self, transport):
        self._transport = transport

    def send(self, path, request):
        try:
            status, response = self._transport.send(path, request)
        except UnicodeError as error:
            raise OSError("Incomplete UTF-8 response") from error
        # A malformed body may follow an executed action. Raise a transport error
        # so v1 retries the SAME ID, then marks the outcome uncertain if exhausted.
        if not isinstance(response, dict) or type(response.get("accepted")) is not bool:
            raise OSError("Incomplete simulator response schema")
        if response["accepted"] and status == 200:
            if not finite(response.get("virtual_time_s")) or response["virtual_time_s"] < 0:
                raise OSError("Invalid simulator clock")
            if not finite(response.get("real_timestamp_ms")):
                raise OSError("Invalid simulator real timestamp")
            if path == "/enter":
                value = response.get("remaining_real_duration_s")
                if not finite(value) or not 0 <= value <= 1200:
                    raise OSError("Invalid enter budget")
            elif path == "/measure":
                kind = response.get("measure_result")
                if kind not in {"direction", "near", "no_signal"}:
                    raise OSError("Invalid measurement result")
                if kind == "direction" and not (finite(response.get("svd_deg")) and 0 <= response["svd_deg"] < 360):
                    raise OSError("Invalid bearing")
            elif path == "/clear" and response.get("clear_result") not in {"success", "no_target_in_range"}:
                raise OSError("Invalid clear result")
            elif path == "/exit" and response.get("exit_reason") != "user_exit":
                raise OSError("Invalid exit result")
        return status, response
