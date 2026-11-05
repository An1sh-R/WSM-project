"""Tiny local web server: serves web/index.html and a JSON search API.

    GET  /api/info                      -> corpus statistics
    GET  /api/search?q=...              -> search results
    POST /api/feedback {query, relevant: [filenames]} -> Rocchio-refined results

Uses only the standard library and listens on localhost only.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MAX_BODY = 64 * 1024


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # keep the console quiet

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self._json({
                    "documents": engine.index.document_count(),
                    "terms": len(engine.index.vocabulary()),
                    "stemming": engine.stemming,
                    "sublinear_tf": engine.sublinear_tf,
                })
            elif url.path == "/api/search":
                query = parse_qs(url.query).get("q", [""])[0]
                self._json(engine.search(query))
            else:
                self._send(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if urlparse(self.path).path != "/api/feedback":
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_BODY:
                    raise ValueError("body too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                query = str(data.get("query", ""))
                relevant = [str(n) for n in data.get("relevant", [])]
            except (ValueError, TypeError):
                self._json({"status": "bad_request", "results": []}, 400)
                return
            self._json(engine.feedback(query, relevant))

    return Handler


def serve(engine, port=8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
    return server
