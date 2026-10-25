"""A local OpenAI-compatible HTTP endpoint with a durable model-request cap.

Point an agent's API base at :attr:`LLMBudgetProxy.base_url`. Every admitted
generation request reserves one call before contacting the upstream, including
requests that fail or are retried by the client. The proxy itself never retries
or follows redirects. The JSONL ledger contains only request metadata, never
credentials, prompts, responses, or query parameters.

This controls HTTP requests routed through the proxy; it is not a sandbox for
an agent that can read credentials and contact a different endpoint itself.
"""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


_GENERATIONS = {"/chat/completions", "/completions", "/responses", "/responses/compact"}
_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
_MAX_BODY = 64 * 1024 * 1024


class _ProxyClosed(Exception):
    pass


class _Budget:
    def __init__(self, path: Path, limit: int):
        self.path, self.limit = path, limit
        self.lock = threading.Lock()
        self.closed = True
        self.generation = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite an existing run's budget when restarting its proxy.
        path.touch(exist_ok=True)

    @staticmethod
    def _records(handle) -> list[dict]:
        records = [json.loads(line) for line in handle if line.strip()]
        if any(not isinstance(r, dict) or r.get("call") != i
               for i, r in enumerate(records, 1)):
            raise ValueError("invalid API call ledger")
        return records

    def count(self) -> int:
        with self.lock, self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            return len(self._records(handle))

    def set_closed(self, closed: bool):
        with self.lock:
            self.closed = closed
            if not closed:
                self.generation += 1
            return self.generation

    def reserve(self, path: str | None, generation: int) -> bool:
        # flock also protects against a second process reopening this ledger.
        with self.lock, self.path.open("r+", encoding="utf-8") as handle:
            if self.closed or generation != self.generation:
                raise _ProxyClosed
            if path is None:  # Metadata requests still need admission while open.
                return True
            fcntl.flock(handle, fcntl.LOCK_EX)
            used = len(self._records(handle))
            if used >= self.limit:
                return False
            record = {"call": used + 1, "time": datetime.now(timezone.utc).isoformat(),
                      "method": "POST", "path": path}
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def handle_error(self, request, client_address):
        # BaseHTTPRequestHandler's default traceback/logging can include data
        # from a malformed request. Errors are reported to the client instead.
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.proxy.timeout)

    def _error(self, status: int, code: str, message: str):
        body = json.dumps({"error": {"type": code, "code": code, "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _body(self) -> bytes:
        length = self.headers.get("Content-Length")
        encoding = self.headers.get("Transfer-Encoding")
        if encoding:
            if length is not None or encoding.lower() != "chunked":
                raise ValueError("unsupported request framing")
            data = bytearray()
            while True:
                line = self.rfile.readline(65537)
                if len(line) > 65536 or not line.endswith(b"\r\n"):
                    raise ValueError("invalid request chunk")
                size = int(line.split(b";", 1)[0].strip(), 16)
                if size < 0 or len(data) + size > _MAX_BODY:
                    raise ValueError("request body exceeds limit")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline(65537)
                        if trailer == b"\r\n":
                            return bytes(data)
                        if len(trailer) > 65536 or not trailer.endswith(b"\r\n"):
                            raise ValueError("invalid request trailer")
                chunk = self.rfile.read(size)
                if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                    raise ValueError("truncated request chunk")
                data.extend(chunk)
        size = int(length or "0")
        if size < 0 or size > _MAX_BODY:
            raise ValueError("request body exceeds limit")
        body = self.rfile.read(size)
        if len(body) != size:
            raise ValueError("truncated request body")
        return body

    def do_POST(self):
        self._forward()

    def do_GET(self):
        self._forward()

    def _forward(self):
        proxy = self.server.proxy
        parsed = urlsplit(self.path)
        # Only origin-form paths under this proxy's base are accepted. Unknown
        # generation endpoints cannot bypass the counter.
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/v1/"):
            self._error(404, "unsupported_endpoint", "Unsupported API endpoint.")
            return
        route = parsed.path[3:]
        generation = self.command == "POST" and route in _GENERATIONS
        metadata = self.command == "GET" and (
            route == "/models" or route.startswith("/models/")
        )
        if self.headers.get("Upgrade"):
            self._error(400, "unsupported_transport", "Use HTTP streaming instead of WebSockets.")
            return
        if not generation and not metadata:
            self._error(404, "unsupported_endpoint", "Unsupported API endpoint.")
            return
        try:
            body = self._body()
        except (ValueError, OSError):
            self._error(400, "invalid_request", "Invalid API request body.")
            return
        try:
            admitted = proxy._budget.reserve(route if generation else None, self.server.generation)
        except _ProxyClosed:
            self._error(503, "proxy_closed", "The API budget proxy has stopped.")
            return
        except (OSError, ValueError):
            self._error(500, "budget_ledger_error", "API budget ledger is unavailable.")
            return
        if not admitted:
            with proxy._rejected_lock:
                proxy._rejected_requests += 1
            self._error(402, "budget_exhausted",
                        f"API call budget exhausted ({proxy.budget} calls). "
                        "Keep the best existing candidate and stop requesting the model.")
            return
        connection_type = (http.client.HTTPSConnection if proxy._upstream.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(proxy._upstream.hostname, proxy._upstream.port,
                                     timeout=proxy.timeout)
        sent_headers = False
        try:
            connection_headers = {h.strip().lower() for h in
                                  self.headers.get("Connection", "").split(",")}
            excluded = _HOP_HEADERS | connection_headers | {"host", "content-length"}
            headers = {key: value for key, value in self.headers.items()
                       if key.lower() not in excluded}
            if proxy._api_key is not None:
                headers = {key: value for key, value in headers.items()
                           if key.lower() != "authorization"}
                headers["Authorization"] = f"Bearer {proxy._api_key}"
            headers["Content-Length"] = str(len(body))
            headers["Connection"] = "close"
            target = proxy._upstream.path.rstrip("/") + route
            if parsed.query:
                target += "?" + parsed.query
            connection.request(self.command, target, body=body, headers=headers)
            upstream = connection.getresponse()
            if 300 <= upstream.status < 400:
                # An absolute Location could make a redirect-following client
                # send its next model request directly to the upstream.
                self._error(502, "upstream_redirect", "Upstream API redirects are not supported.")
                return
            self.send_response(upstream.status)
            connection_headers = {h.strip().lower() for h in
                                  upstream.getheader("Connection", "").split(",")}
            excluded = _HOP_HEADERS | connection_headers | {"content-length"}
            for key, value in upstream.getheaders():
                if key.lower() not in excluded:
                    self.send_header(key, value)
            # HTTPResponse decodes upstream chunk framing. A close-delimited
            # response preserves SSE streaming without buffering or re-encoding.
            self.send_header("Connection", "close")
            self.end_headers()
            sent_headers = True
            while chunk := upstream.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not sent_headers:
                self._error(502, "upstream_error", "The upstream API request failed.")
        finally:
            connection.close()
            self.close_connection = True


class LLMBudgetProxy:
    """Meter model requests from one agent/case without changing its LLM loop.

    ``ledger_path`` is a dedicated JSONL file, separate from agent usage events.
    A request is charged before its upstream attempt and is never refunded,
    even on an upstream error or client disconnect. Existing records are loaded
    on restart. Only supported generation routes and model metadata GETs are
    forwarded; metadata GETs do not consume the model-generation budget.

    ``api_key`` optionally supplies the upstream Bearer credential; otherwise
    the caller's Authorization header is preserved. HTTP streaming is supported,
    WebSockets are explicitly rejected. ``start``/``close`` and ``with`` are both
    supported; closing the proxy never initiates another upstream request.
    """

    def __init__(self, upstream_base_url: str, budget: int, ledger_path: str | Path,
                 *, api_key: str | None = None, timeout: float = 600):
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("API call budget must be a positive integer")
        upstream = urlsplit(upstream_base_url)
        if (upstream.scheme not in {"http", "https"} or not upstream.hostname
                or upstream.username or upstream.password or upstream.query or upstream.fragment):
            raise ValueError("upstream API base must be an HTTP(S) URL without credentials or query")
        if timeout <= 0:
            raise ValueError("API timeout must be positive")
        self.budget, self.timeout = budget, timeout
        self._upstream, self._api_key = upstream, api_key
        self._budget = _Budget(Path(ledger_path), budget)
        self._budget.count()  # Fail closed before launching an agent on a corrupt ledger.
        self._server = None
        self._thread = None
        self._rejected_lock = threading.Lock()
        self._rejected_requests = 0

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("API budget proxy is not running")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def calls_used(self) -> int:
        return self._budget.count()

    @property
    def exhausted(self) -> bool:
        return self.calls_used >= self.budget

    @property
    def rejected_requests(self) -> int:
        with self._rejected_lock:
            return self._rejected_requests

    def start(self) -> "LLMBudgetProxy":
        if self._server is None:
            self._server = _Server(("127.0.0.1", 0), _Handler)
            self._server.proxy = self
            self._server.generation = self._budget.set_closed(False)
            self._thread = threading.Thread(target=self._server.serve_forever,
                                            kwargs={"poll_interval": 0.05}, daemon=True)
            self._thread.start()
        return self

    def close(self):
        # Freeze admission under the reservation lock before stopping accept().
        # Already admitted requests may finish, but a slow request body cannot
        # consume another call after close() has taken the final ledger snapshot.
        self._budget.set_closed(True)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = self._thread = None

    stop = close

    def __enter__(self) -> "LLMBudgetProxy":
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
