"""The API cap is checked against a local fake provider, without real model calls."""

import concurrent.futures
import gzip
import http.client
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from evograd.adbench_tools.llm_budget import LLMBudgetProxy


class _Upstream(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


@pytest.fixture
def upstream():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.do_POST()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with self.server.lock:
                self.server.received.append({"path": self.path, "body": body,
                                             "authorization": self.headers.get("Authorization"),
                                             "content_encoding": self.headers.get("Content-Encoding")})
            if "stream" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in (b'data: {"part":1}\n\n', b'data: [DONE]\n\n'):
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    if b"part" in chunk:
                        self.server.release_stream.wait(5)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                return
            if "redirect" in self.path:
                status, response = 307, b""
            elif "fail" in self.path:
                status, response = 500, b'{"error":"retry this failure"}'
            else:
                status, response = 200, b'{"ok":true}'
            self.send_response(status)
            if status == 307:
                self.send_header("Location", "/provider/v1/responses")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = _Upstream(("127.0.0.1", 0), Handler)
    server.received, server.lock = [], threading.Lock()
    server.release_stream = threading.Event()
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    server.base_url = f"http://127.0.0.1:{server.server_port}/provider/v1"
    try:
        yield server
    finally:
        server.release_stream.set()
        server.shutdown()
        server.server_close()
        thread.join()


def request(base_url, route="/responses", *, method="POST", body=b'{"model":"fake"}',
            headers=None, chunked=False):
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        connection.request(method, parsed.path + route, body=body,
                           headers=headers or {}, encode_chunked=chunked)
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


@pytest.mark.parametrize("route", ["/chat/completions", "/completions", "/responses",
                                  "/responses/compact"])
def test_generation_routes_are_metered_and_cap_is_not_retryable(upstream, tmp_path, route):
    path = tmp_path / "api_calls.jsonl"
    with LLMBudgetProxy(upstream.base_url, 1, path, api_key="test-upstream-secret") as proxy:
        assert request(proxy.base_url, route + "?api_key=query-secret", body=b"private prompt",
                       headers={"Authorization": "Bearer client-secret"})[0] == 200
        status, body, _ = request(proxy.base_url, route)
        assert status == 402
        assert json.loads(body)["error"]["code"] == "budget_exhausted"
        assert proxy.calls_used == 1 and proxy.exhausted and proxy.rejected_requests == 1
    assert len(upstream.received) == 1
    assert upstream.received[0]["path"] == "/provider/v1" + route + "?api_key=query-secret"
    assert upstream.received[0]["body"] == b"private prompt"
    assert upstream.received[0]["authorization"] == "Bearer test-upstream-secret"
    record = json.loads(path.read_text())
    assert record["call"] == 1 and record["path"] == route
    assert not any(word in path.read_text() for word in
                   ("test-upstream-secret", "client-secret", "query-secret", "private prompt"))


def test_concurrent_proxies_share_the_durable_limit(upstream, tmp_path):
    ledger = tmp_path / "api_calls.jsonl"
    with LLMBudgetProxy(upstream.base_url, 10, ledger) as first, \
            LLMBudgetProxy(upstream.base_url, 10, ledger) as second:
        bases = [first.base_url, second.base_url] * 12
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(request, bases))
        assert sum(status == 200 for status, _, _ in results) == 10
        assert sum(status == 402 for status, _, _ in results) == 14
        assert first.calls_used == second.calls_used == 10
        assert first.rejected_requests + second.rejected_requests == 14
    assert len(upstream.received) == 10
    assert [json.loads(line)["call"] for line in ledger.read_text().splitlines()] == list(range(1, 11))
    with LLMBudgetProxy(upstream.base_url, 10, ledger) as resumed:
        assert request(resumed.base_url)[0] == 402
        assert resumed.calls_used == 10
    assert len(upstream.received) == 10


def test_failed_requests_and_client_retries_each_cost_a_call(upstream, tmp_path):
    with LLMBudgetProxy(upstream.base_url, 2, tmp_path / "calls.jsonl") as proxy:
        assert request(proxy.base_url, "/chat/completions?fail=1")[0] == 500
        assert request(proxy.base_url, "/chat/completions?fail=1")[0] == 500
        assert request(proxy.base_url, "/chat/completions?fail=1")[0] == 402
        assert proxy.calls_used == 2
    assert len(upstream.received) == 2


def test_sse_is_forwarded_before_the_upstream_finishes(upstream, tmp_path):
    with LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl") as proxy:
        parsed = urlsplit(proxy.base_url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
        try:
            connection.request("POST", "/v1/responses?stream=1", body=b"{}")
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == "text/event-stream"
            assert response.readline() == b'data: {"part":1}\n'
            assert not upstream.release_stream.is_set()
            upstream.release_stream.set()
            assert response.read() == b'\ndata: [DONE]\n\n'
        finally:
            upstream.release_stream.set()
            connection.close()
        assert proxy.calls_used == 1


def test_unknown_generation_routes_and_websockets_do_not_reach_upstream(upstream, tmp_path):
    with LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl") as proxy:
        assert request(proxy.base_url, "/embeddings")[0] == 404
        assert request(proxy.base_url, "/responses", method="GET",
                       headers={"Upgrade": "websocket"})[0] == 400
        assert proxy.calls_used == 0
        assert request(proxy.base_url, "/models", method="GET", body=b"")[0] == 200
        assert request(proxy.base_url, "/models/fake", method="GET", body=b"")[0] == 200
        assert proxy.calls_used == 0
    assert [r["path"] for r in upstream.received] == ["/provider/v1/models", "/provider/v1/models/fake"]


def test_redirects_cannot_move_a_client_outside_the_meter(upstream, tmp_path):
    with LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl") as proxy:
        status, body, headers = request(proxy.base_url, "/responses?redirect=1")
        assert status == 502 and "Location" not in headers
        assert json.loads(body)["error"]["code"] == "upstream_redirect"
        assert proxy.calls_used == 1
    assert len(upstream.received) == 1


def test_chunked_request_and_client_authorization_are_preserved(upstream, tmp_path):
    with LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl") as proxy:
        assert request(proxy.base_url, body=iter([b"first", b"second"]), chunked=True,
                       headers={"Authorization": "Bearer fake-client-key"})[0] == 200
    assert upstream.received[0]["body"] == b"firstsecond"
    assert upstream.received[0]["authorization"] == "Bearer fake-client-key"


def test_compressed_chunked_body_is_forwarded_without_decoding(upstream, tmp_path):
    compressed = gzip.compress(b'{"model":"fake","input":"a prompt"}')
    with LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl") as proxy:
        assert request(proxy.base_url, body=iter([compressed[:10], compressed[10:]]),
                       chunked=True, headers={"Content-Encoding": "gzip"})[0] == 200
        assert proxy.calls_used == 1
    assert upstream.received[0]["content_encoding"] == "gzip"
    assert upstream.received[0]["body"] == compressed


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "CONNECT"])
def test_unknown_methods_are_not_forwarded(upstream, tmp_path, method):
    with LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl") as proxy:
        assert request(proxy.base_url, method=method)[0] == 501
        assert proxy.calls_used == 0
    assert not upstream.received


def test_openai_sdk_retries_use_the_cap_and_stop_on_402(upstream, tmp_path):
    openai = pytest.importorskip("openai")
    with LLMBudgetProxy(upstream.base_url, 2, tmp_path / "calls.jsonl") as proxy:
        with openai.OpenAI(api_key="fake-key", base_url=proxy.base_url, max_retries=5) as client:
            with pytest.raises(openai.APIStatusError) as error:
                client.post("/responses?fail=1", body={"model": "fake"}, cast_to=dict)
        assert error.value.status_code == 402
        assert error.value.code == "budget_exhausted"
        assert proxy.calls_used == 2 and proxy.rejected_requests == 1
    assert len(upstream.received) == 2


def test_invalid_ledger_fails_closed_before_listening(upstream, tmp_path):
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text('{"call":2}\n')
    with pytest.raises(ValueError, match="invalid API call ledger"):
        LLMBudgetProxy(upstream.base_url, 10, ledger)
    assert not upstream.received


def test_start_stop_are_idempotent_and_restart_keeps_the_count(upstream, tmp_path):
    proxy = LLMBudgetProxy(upstream.base_url, 2, tmp_path / "calls.jsonl")
    with pytest.raises(RuntimeError, match="not running"):
        _ = proxy.base_url
    try:
        assert proxy.start().start() is proxy
        assert request(proxy.base_url)[0] == 200
        proxy.close()
        proxy.stop()
        proxy.start()
        assert request(proxy.base_url)[0] == 200
        assert request(proxy.base_url)[0] == 402
    finally:
        proxy.close()


@pytest.mark.parametrize("restart", [False, True])
def test_close_freezes_admission_for_an_already_accepted_slow_body(upstream, tmp_path, restart):
    proxy = LLMBudgetProxy(upstream.base_url, 1, tmp_path / "calls.jsonl").start()
    parsed = urlsplit(proxy.base_url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=2) as connection:
            connection.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n"
                               b"Content-Length: 2\r\nExpect: 100-continue\r\n\r\n")
            with connection.makefile("rb") as response:
                assert response.readline() == b"HTTP/1.1 100 Continue\r\n"
                assert response.readline() == b"\r\n"
                proxy.close()
                if restart:
                    proxy.start()
                connection.sendall(b"{}")
                assert response.readline() == b"HTTP/1.1 503 Service Unavailable\r\n"
        assert proxy.calls_used == 0 and not upstream.received
    finally:
        proxy.close()
