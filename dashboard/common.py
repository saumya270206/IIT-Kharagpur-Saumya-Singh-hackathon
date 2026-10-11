"""Helpers shared by the dashboard pages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from risk_engine.config import API_HOST, API_PORT  # noqa: E402

API = f"http://{API_HOST}:{API_PORT}"

STATUS_COLOR = {"GREEN": "#2ca02c", "AMBER": "#ff9f1c", "RED": "#d62728"}


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params or None, timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def api_post(path: str, payload: dict, timeout: int = 180) -> dict:
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        body = r.json() if r.content else {}
        return body if r.ok else {"error": f"API {r.status_code}: {body}"}
    except requests.RequestException as exc:
        return {"error": f"API unreachable: {exc}"}


def money(x: float, signed: bool = False) -> str:
    sign = ("+" if x >= 0 else "-") if signed else ("-" if x < 0 else "")
    return f"{sign}${abs(x) / 1e6:,.1f}M"
