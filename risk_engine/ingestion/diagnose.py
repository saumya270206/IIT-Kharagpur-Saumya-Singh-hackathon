"""`python run_engine.py diagnose` - test every live feed and say what is wrong.

Does one raw request per feed (no retries) and prints DNS, TLS, HTTP status,
content type, a body snippet and how many headlines were parsed.
"""

from __future__ import annotations

import socket
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests

from risk_engine import config
from risk_engine.ingestion.gdelt import parse_gdelt
from risk_engine.ingestion.http import explain, verify_setting
from risk_engine.ingestion.newsapi import parse_newsapi
from risk_engine.ingestion.rss import parse_rss


def _targets():
    t = [
        ("gdelt", config.GDELT_URL,
         {"query": config.GDELT_QUERY, "mode": "artlist", "format": "json",
          "maxrecords": "5", "timespan": config.GDELT_TIMESPAN, "sort": "DateDesc"}, {}),
        ("rss (Google News)", config.GOOGLE_NEWS_URL,
         {"q": config.RSS_QUERY, "hl": "en-US", "gl": "US", "ceid": "US:en"}, {}),
        ("yahoo (AAPL)", config.YAHOO_RSS_URL,
         {"s": "AAPL", "region": "US", "lang": "en-US"}, {}),
    ]
    if config.NEWSAPI_KEY:
        t.append(("newsapi", config.NEWSAPI_URL,
                  {"q": config.NEWSAPI_QUERY, "language": "en", "pageSize": "5"},
                  {"X-Api-Key": config.NEWSAPI_KEY}))
    return t


def _parse(label: str, resp: requests.Response) -> tuple[int, str | None]:
    """-> (headline count, problem or None)"""
    try:
        if label.startswith("gdelt"):
            try:
                payload = resp.json()
            except ValueError:
                return 0, f"body is text, not JSON: {resp.text.strip()[:150]!r}"
            return len(parse_gdelt(payload)), None
        if label == "newsapi":
            payload = resp.json()
            if payload.get("status") == "error":
                return 0, f"{payload.get('code')}: {payload.get('message')}"
            return len(parse_newsapi(payload)), None
        return len(parse_rss(resp.content, "diag", 50)), None
    except ET.ParseError as exc:
        return 0, f"invalid XML: {exc}"


def run_diagnostics() -> int:
    print("== Risk engine live-feed diagnostics ==")
    print(f"python {sys.version.split()[0]} | requests {requests.__version__}")
    import os
    proxies = {k: os.environ[k] for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                                          "https_proxy", "NO_PROXY") if k in os.environ}
    print(f"proxy env : {proxies or 'none'}")
    print(f"TLS verify: {verify_setting()}\n")

    working = 0
    for label, url, params, headers in _targets():
        host = urlparse(url).hostname
        print(f"-- {label}  ({host})")
        try:
            addr = socket.getaddrinfo(host, 443)[0][4][0]
            print(f"   DNS     : ok ({addr})")
        except OSError as exc:
            print(f"   DNS     : FAILED ({exc}) -> no internet / DNS blocked / VPN needed")
            continue
        try:
            resp = requests.get(url, params=params, timeout=20, verify=verify_setting(),
                                headers={"User-Agent": config.HTTP_USER_AGENT, **headers})
        except requests.exceptions.RequestException as exc:
            print(f"   REQUEST : FAILED -> {explain(exc)}")
            continue
        print(f"   HTTP    : {resp.status_code}  content-type={resp.headers.get('content-type')}"
              f"  bytes={len(resp.content)}")
        if resp.status_code == 429:
            print("   -> rate limited; wait ~10 s and retry (GDELT allows 1 request / 5 s)")
            continue
        if resp.status_code >= 400:
            print(f"   -> body: {resp.text.strip()[:150]!r}")
            continue
        n, problem = _parse(label, resp)
        if problem:
            print(f"   PARSE   : FAILED -> {problem}")
        elif n == 0:
            print("   PARSE   : ok but 0 headlines (query/timespan too narrow?)")
        else:
            print(f"   PARSE   : ok - {n} headlines  =>  this feed WORKS")
            working += 1

    print()
    if working:
        print(f"RESULT: {working} feed(s) working. `python run_engine.py once --source live` will succeed.")
        return 0
    print("RESULT: no live feed reachable from this machine. Fix the cause above "
          "(network/VPN/proxy/TLS), or run offline with `--source datasets`.")
    return 1