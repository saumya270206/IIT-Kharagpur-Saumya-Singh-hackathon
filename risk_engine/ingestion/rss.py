"""Live news ingestion from RSS feeds (free, no API key).

* Google News RSS - general financial headlines (``fetch_rss``)
* Yahoo Finance per-ticker RSS - company headlines (``fetch_yahoo``)

They are the fallbacks when GDELT is rate-limited or unreachable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from risk_engine import config
from risk_engine.ingestion.http import FetchError, http_get, record
from risk_engine.models import RawItem

log = logging.getLogger("risk_engine.rss")

_SUFFIX_RE = re.compile(r"\s+[-|\u2013\u2014]\s+[^-|\u2013\u2014]{2,40}$")


def _sid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def clean_title(title: str, publisher: str | None = None) -> str:
    """Drop the trailing ' - Publisher' that Google News appends to headlines."""
    title = re.sub(r"\s+", " ", title).strip()
    if publisher and title.endswith(publisher):
        stripped = title[: -len(publisher)].rstrip(" -|\u2013\u2014").strip()
        if stripped:
            return stripped
    m = _SUFFIX_RE.search(title)
    return title[: m.start()].strip() if m and m.start() > 15 else title


def parse_rss(content: bytes, prefix: str, limit: int,
              ticker: str | None = None) -> list[RawItem]:
    root = ET.fromstring(content)
    items: list[RawItem] = []
    for node in root.findall("./channel/item"):
        if len(items) >= limit:
            break
        raw_title = (node.findtext("title") or "").strip()
        if not raw_title:
            continue
        link = (node.findtext("link") or "").strip() or None
        title = clean_title(raw_title, (node.findtext("source") or "").strip() or None)
        items.append(
            RawItem(
                id=f"{prefix}-{_sid(link or title)}",
                source="news",
                text=title,
                url=link,
                published_at=_parse_pubdate(node.findtext("pubDate")),
                ticker=ticker,
            )
        )
    return items


def fetch_rss(query: str | None = None, max_records: int | None = None) -> list[RawItem]:
    """Fetch recent financial headlines from Google News RSS."""
    limit = max_records or config.GDELT_MAX_RECORDS
    params = {"q": query or config.RSS_QUERY, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        resp = http_get(config.GOOGLE_NEWS_URL, params=params, label="rss",
                        timeout=config.RSS_TIMEOUT)
        items = parse_rss(resp.content, "rss", limit)
    except FetchError as exc:
        record("rss", 0, str(exc))
        return []
    except ET.ParseError as exc:
        record("rss", 0, f"Google News did not return valid RSS/XML ({exc})")
        return []
    record("rss", len(items), None if items else "Google News returned no items")
    return items


def fetch_yahoo(tickers: list[str] | None = None, per_ticker: int | None = None) -> list[RawItem]:
    """Per-ticker headlines from the Yahoo Finance RSS feed."""
    tickers = tickers or config.YAHOO_TICKERS
    per = per_ticker or config.YAHOO_PER_TICKER
    items: list[RawItem] = []
    failures, last_err = 0, None
    for t in tickers:
        try:
            resp = http_get(config.YAHOO_RSS_URL,
                            params={"s": t, "region": "US", "lang": "en-US"},
                            label="yahoo", timeout=config.RSS_TIMEOUT, retries=0)
            items += parse_rss(resp.content, f"yahoo-{t.lower()}", per, ticker=t)
            failures = 0
        except (FetchError, ET.ParseError) as exc:
            failures, last_err = failures + 1, str(exc)
            if failures >= 3:        # feed is down / blocking us: stop hammering it
                break
        time.sleep(0.2)
    err = None if items else (last_err or "Yahoo Finance RSS returned no items")
    record("yahoo", len(items), err)
    return items


def _parse_pubdate(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = parsedate_to_datetime(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)