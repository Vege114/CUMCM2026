"""Local HTTP boundary and Windows-facing CLI without the exe or formal tests."""
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.baseline_v1.baseline.mock import MockTransport
from experiments.baseline_v2.run import main
from experiments.baseline_v2.solver.protocol import CheckedTransport, Client, HttpTransport


class HttpTests(unittest.TestCase):
    def test_real_loopback_json_four_actions_and_clear_channel(self):
        engine = MockTransport([])
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.requests.append((self.path, self.headers.get("Content-Type")))
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status, data = engine.send(self.path, request)
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        server.requests = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client = Client(CheckedTransport(HttpTransport(f"http://127.0.0.1:{server.server_port}")),
                                "OFFLINE", Path(tmp)/"events.jsonl")
                try:
                    client.call("/enter")
                    client.call("/measure", (300, 400), 1)
                    client.call("/measure", (300, 400), 2)
                    client.call("/clear", (300, 0), 3)
                    client.call("/measure", (300, 0), 2)
                    self.assertEqual(client.call("/exit")["virtual_time_s"], 199)
                finally:
                    client.close()
            self.assertEqual({p for p, _ in server.requests}, {"/enter", "/measure", "/clear", "/exit"})
            self.assertTrue(all(content == "application/json" for _, content in server.requests))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_official_dry_run_sends_zero_requests(self):
        buffer = io.StringIO()
        with patch.object(HttpTransport, "send", side_effect=AssertionError("HTTP forbidden")), redirect_stdout(buffer):
            code = main(["--backend", "official", "--mode", "practice", "--case-code", "UI-CODE", "--problem", "4", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue())["http_requests_sent"], 0)

    def test_official_run_rejected_on_macos_before_http(self):
        with patch("sys.platform", "darwin"), redirect_stderr(io.StringIO()), patch.object(HttpTransport, "send", side_effect=AssertionError("HTTP forbidden")):
            with self.assertRaises(SystemExit) as error:
                main(["--backend", "official", "--mode", "practice", "--case-code", "UI-CODE"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
