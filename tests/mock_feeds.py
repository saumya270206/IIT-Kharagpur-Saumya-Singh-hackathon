"""A local HTTP server imitating GDELT, Google News RSS, Yahoo Finance RSS and NewsAPI.

Used by tests and for an offline end-to-end run of the *live* code path:

    python tests/mock_feeds.py            # serves on 127.0.0.1:8765
    GDELT_URL=http://127.0.0.1:8765/gdelt GOOGLE_NEWS_URL=http://127.0.0.1:8765/gnews \\
    YAHOO_RSS_URL=http://127.0.0.1:8765/yahoo python run_engine.py once --source live

``mode`` per feed can be switched at runtime via ``/_mode?feed=gdelt&mode=429_once``.
Modes: ok | 429_once | text_error | down (500) | empty
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

NOW = datetime.now(timezone.utc)

GDELT_ARTICLES = [
    ("Nvidia shares surge after record data-center revenue beats estimates", 30),
    ("Tesla recalls 1.2 million vehicles over brake defect, probe widens", 55),
    ("JPMorgan downgraded by Moody's as credit losses mount, shares plunge", 80),
    ("Oil jumps as Middle East conflict escalates and sanctions loom", 95),
    ("Microsoft to acquire cybersecurity startup in $4 billion deal", 120),
    ("Fed signals rate cut as inflation cools, markets rally", 150),
]
GNEWS_ITEMS = [
    ("Apple unveils new iPhone with record preorders", "Reuters", 20),
    ("Amazon Web Services outage knocks major websites offline", "Bloomberg", 45),
    ("UnitedHealth lowers full-year guidance as medical costs rise", "CNBC", 70),
    ("Meta faces EU antitrust probe over advertising practices", "Financial Times", 100),
]
YAHOO_ITEMS = {
    "AAPL": [("Apple stock climbs on strong services growth", 15)],
    "XOM": [("Exxon Mobil raises dividend as oil prices rally", 25)],
}

STATE = {"gdelt": "ok", "gnews": "ok", "yahoo": "ok", "newsapi": "ok", "_hits": {}}


def _gdelt_payload() -> str:
    arts = [{
        "url": f"https://news.example.com/{i}/{t[:20].replace(' ', '-').lower()}",
        "url_mobile": "", "title": t, "seendate": (NOW - timedelta(minutes=m)).strftime("%Y%m%dT%H%M%SZ"),
        "socialimage": "", "domain": "news.example.com", "language": "English",
        "sourcecountry": "United States",
    } for i, (t, m) in enumerate(GDELT_ARTICLES)]
    return json.dumps({"articles": arts})


def _rss(items: list[tuple[str, str | None, int]]) -> str:
    body = ""
    for title, src, mins in items:
        full = f"{title} - {src}" if src else title
        body += (f"<item><title>{full}</title><link>https://example.com/{abs(hash(title)) % 10**6}</link>"
                 f"<pubDate>{format_datetime(NOW - timedelta(minutes=mins))}</pubDate>"
                 + (f"<source url='https://x.com'>{src}</source>" if src else "") + "</item>")
    return f"<?xml version='1.0'?><rss version='2.0'><channel><title>t</title>{body}</channel></rss>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep test output clean
        pass

    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path == "/_mode":
            STATE[q["feed"][0]] = q["mode"][0]
            STATE["_hits"] = {}
            return self._send(200, "ok", "text/plain")
        feed = {"/gdelt": "gdelt", "/gnews": "gnews", "/yahoo": "yahoo", "/newsapi": "newsapi"}.get(path)
        if feed is None:
            return self._send(404, "not found", "text/plain")
        STATE["_hits"][feed] = STATE["_hits"].get(feed, 0) + 1
        mode = STATE[feed]
        if mode == "down":
            return self._send(500, "boom", "text/plain")
        if mode == "429_once" and STATE["_hits"][feed] == 1:
            return self._send(429, "Please limit requests to one every 5 seconds.", "text/html")
        if mode == "text_error":
            return self._send(200, "The specified phrase is too short.", "text/html; charset=utf-8")
        if feed == "gdelt":
            return self._send(200, '{"articles": []}' if mode == "empty" else _gdelt_payload(),
                              "application/json")
        if feed == "gnews":
            return self._send(200, _rss([] if mode == "empty" else GNEWS_ITEMS), "application/rss+xml")
        if feed == "yahoo":
            t = q.get("s", ["AAPL"])[0]
            items = [] if mode == "empty" else [(x, None, m) for x, m in YAHOO_ITEMS.get(t, [])]
            return self._send(200, _rss(items), "text/xml")
        if feed == "newsapi":
            return self._send(200, json.dumps({"status": "ok", "articles": [
                {"title": "AMD plunges on weak guidance", "description": "Analysts turn bearish.",
                 "url": "https://example.com/amd", "publishedAt": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}]}),
                "application/json")


def start(port: int = 0) -> tuple[HTTPServer, str]:
    for k in ("gdelt", "gnews", "yahoo", "newsapi"):
        STATE[k] = "ok"
    STATE["_hits"] = {}
    srv = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


if __name__ == "__main__":
    srv, base = start(8765)
    print(f"mock feeds on {base}  (/gdelt /gnews /yahoo /newsapi)")
    threading.Event().wait()