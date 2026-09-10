"""Serial HTTP/JSON transport, exact idempotent retries and redacted event journal."""
import copy
import http.client
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener


class ProtocolError(RuntimeError):
    pass


class TransportUnavailable(ProtocolError):
    pass


class HttpTransport:
    def __init__(self, base_url: str, timeout_s: float = 5):
        url = urlsplit(base_url)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Simulator transport is restricted to local loopback HTTP")
        if url.path not in ("", "/") or url.query or url.fragment or url.username:
            raise ValueError("Expected a plain loopback base URL")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.opener = build_opener(ProxyHandler({}))

    def send(self, path: str, request: dict) -> tuple[int, dict]:
        data = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8")
        req = Request(self.base_url+path, data=data, method="POST",
                      headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout_s) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))


class Client:
    def __init__(self, transport, robot_id: str, events_path: Path, retries: int = 2):
        self.transport, self.robot_id, self.retries = transport, robot_id, retries
        self.session_id = uuid.uuid4().hex[:12]
        self.sequence = 0
        self.started = time.monotonic()
        self.virtual_time_s = 0.0
        self.active = False
        self.uncertain = False
        self.journal = events_path.open("w", encoding="utf-8")

    def journal_event(self, path, request, response, status, **extra):
        safe_request = copy.deepcopy(request)
        safe_request["robot_id"] = "REDACTED"
        event = {"path": path, "request": safe_request, "response": response,
                 "http_status": status, "client_elapsed_s": time.monotonic()-self.started, **extra}
        self.journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
        self.journal.flush()

    def call(self, path: str, position=None, channel=None) -> dict:
        self.sequence += 1
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": f"{self.session_id}-{self.sequence}"}
        if position is not None:
            payload.update(position={"x": position[0], "y": position[1]}, channel=channel)
        for attempt in range(self.retries+1):
            try:
                status, response = self.transport.send(path, payload)
            except (OSError, URLError, http.client.HTTPException, json.JSONDecodeError) as error:
                self.journal_event(path, payload, None, None, attempt=attempt,
                                   transport_error=type(error).__name__)
                if attempt == self.retries:
                    self.uncertain = True
                    raise TransportUnavailable("No complete response; last action outcome unknown") from error
                time.sleep(0.25*(2**attempt))
                continue
            self.journal_event(path, payload, response, status, attempt=attempt)
            if status != 200 or response.get("accepted") is not True:
                raise ProtocolError(f"{path}: HTTP {status}, accepted={response.get('accepted')}")
            self.virtual_time_s = float(response["virtual_time_s"])
            if path == "/enter":
                self.active = True
            elif path == "/exit":
                self.active = False
            return response
        raise AssertionError("unreachable")

    def close(self):
        self.journal.close()
