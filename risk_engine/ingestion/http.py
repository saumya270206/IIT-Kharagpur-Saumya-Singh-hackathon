"""Shared, resilient HTTP GET for the live feeds.

* browser-like User-Agent (some feeds reject the default python-requests one)
* retries with exponential back-off on timeouts, connection errors, 429 and 5xx
* honours the standard HTTPS_PROXY / HTTP_PROXY env vars (requests does this)
* optional custom CA bundle / SSL-verify switch for corporate networks
* raises ``FetchError`` with a *human readable* reason instead of a stack trace
* records the outcome of every feed in ``STATUS`` so callers can report it
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from risk_engine import config

log = logging.getLogger("risk_engine.http")

# feed label -> {"ok": bool, "count": int, "error": str | None}
STATUS: dict[str, dict] = {}


class FetchError(Exception):
    """A feed could not be fetched; the message explains why."""


def verify_setting():
    if config.CA_BUNDLE:
        return config.CA_BUNDLE
    return config.SSL_VERIFY


def explain(exc: Exception) -> str:
    """Translate a requests exception into an actionable one-liner."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, requests.exceptions.SSLError) or "certificate" in low:
        return ("TLS/certificate error (corporate proxy or antivirus intercepting HTTPS?) "
                "- set RISK_CA_BUNDLE=<path to your CA .pem>, or RISK_SSL_VERIFY=0 as a last resort")
    if isinstance(exc, requests.exceptions.ProxyError):
        return "proxy error - check HTTPS_PROXY / HTTP_PROXY"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timed out - slow network or firewall blocking the host"
    if isinstance(exc, requests.exceptions.ConnectionError):
        if any(k in low for k in ("name resolution", "getaddrinfo", "name or service",
                                  "nodename", "failed to resolve")):
            return "DNS lookup failed - no internet access, VPN/proxy needed, or host blocked"
        return f"connection failed ({msg[:120]})"
    return msg[:200]


def http_get(
    url: str,
    *,
    label: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 15.0,
    retries: Optional[int] = None,
    rate_limit_wait: float = 6.0,
) -> requests.Response:
    """GET ``url`` or raise FetchError. 4xx (except 429) is not retried."""
    attempts = 1 + (config.HTTP_RETRIES if retries is None else retries)
    hdrs = {"User-Agent": config.HTTP_USER_AGENT, "Accept": "*/*", **(headers or {})}
    last = "unknown error"
    for n in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout,
                                verify=verify_setting())
        except requests.exceptions.RequestException as exc:
            last = explain(exc)
            if isinstance(exc, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
                break  # retrying will not help
        else:
            if resp.status_code == 429:
                last = "HTTP 429 rate limited"
                wait = rate_limit_wait
                try:
                    wait = max(wait, float(resp.headers.get("Retry-After", 0)))
                except ValueError:
                    pass
                if n < attempts - 1:
                    log.info("%s: rate limited, waiting %.0fs", label, wait)
                    time.sleep(wait)
                continue
            if resp.status_code >= 500:
                last = f"HTTP {resp.status_code} server error"
            elif resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code} from {label}")
            else:
                return resp
        if n < attempts - 1:
            time.sleep(config.HTTP_BACKOFF * (2 ** n))
    raise FetchError(last)


def record(label: str, count: int, error: Optional[str] = None) -> None:
    STATUS[label] = {"ok": error is None, "count": count, "error": error}
    if error:
        log.warning("%s feed unavailable: %s", label, error)