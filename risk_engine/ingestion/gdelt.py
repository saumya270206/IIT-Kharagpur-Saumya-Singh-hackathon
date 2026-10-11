"""Live news ingestion from The GDELT Project (free, no API key)."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timezone

from risk_engine import config
from risk_engine.ingestion.http import FetchError, http_get, record
from risk_engine.models import RawItem

log = logging.getLogger("risk_engine.gdelt")


def _sid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def parse_gdelt(payload: dict) -> list[RawItem]:
    items: list[RawItem] = []
    for art in payload.get("articles", []) or []:
        title = (art.get("title") or "").strip()
        if not title:
            continue
        url = art.get("url")
        items.append(
            RawItem(
                id=f"gdelt-{_sid(url or title)}",
                source="news",
                text=title,  # GDELT exposes headlines; the model scores the headline
                url=url,
                published_at=_parse_seendate(art.get("seendate")),
            )
        )
    return items


def _fetch_one(query: str, max_records: int, timespan: str | None) -> list[RawItem]:
    """One GDELT request. Raises FetchError with a readable reason."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "timespan": timespan or config.GDELT_TIMESPAN,
        "sort": "DateDesc",
    }
    resp = http_get(config.GDELT_URL, params=params, label="gdelt",
                    timeout=config.GDELT_TIMEOUT, rate_limit_wait=6.0)
    try:
        payload = resp.json()
    except ValueError:
        # GDELT reports bad queries as plain text with HTTP 200
        raise FetchError(f"GDELT answered with text, not JSON: {resp.text.strip()[:160]!r}")
    return parse_gdelt(payload)


def fetch_gdelt(
    query: str | None = None,
    max_records: int | None = None,
    timespan: str | None = None,
) -> list[RawItem]:
    """Fetch recent headlines from the GDELT DOC 2.0 API.

    Without ``query`` the short per-batch queries in ``config.GDELT_QUERIES`` run one
    after another (paused to respect GDELT's 1 request / 5 s limit) and are merged
    and de-duplicated. A batch whose *query* GDELT rejects is skipped; any other
    failure (network, 429, 5xx) stops the run early.

    Returns [] (and records the reason in ``http.STATUS['gdelt']``) on failure so a
    demo never breaks when offline.
    """
    queries = [query] if query else list(config.GDELT_QUERIES)
    total = max_records or config.GDELT_MAX_RECORDS
    per_query = total if len(queries) == 1 else max(5, math.ceil(total / len(queries)))

    items: dict[str, RawItem] = {}
    last_error: str | None = None
    for n, q in enumerate(queries):
        if n:
            time.sleep(config.GDELT_BATCH_PAUSE)
        try:
            batch = _fetch_one(q, per_query, timespan)
        except FetchError as exc:
            last_error = str(exc)
            if last_error.startswith("GDELT answered with text"):
                log.warning("gdelt: query %d/%d rejected (%s)", n + 1, len(queries), last_error)
                continue          # bad query only - the other batches may still work
            break                 # network / rate limit / server error: do not hammer
        for it in batch:
            items.setdefault(it.id, it)

    out = list(items.values())
    if out:
        record("gdelt", len(out))
    else:
        record("gdelt", 0, last_error or "GDELT returned no articles for the query/timespan")
    return out


def _parse_seendate(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)