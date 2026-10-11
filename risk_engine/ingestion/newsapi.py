"""Optional live source: NewsAPI.org (free developer key in NEWSAPI_KEY)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from risk_engine import config
from risk_engine.ingestion.http import FetchError, http_get, record
from risk_engine.models import RawItem

log = logging.getLogger("risk_engine.newsapi")


def parse_newsapi(payload: dict) -> list[RawItem]:
    items: list[RawItem] = []
    for art in payload.get("articles", []) or []:
        title = (art.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        desc = (art.get("description") or "").strip()
        url = art.get("url")
        try:
            ts = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
        except (KeyError, ValueError, AttributeError):
            ts = datetime.now(timezone.utc)
        items.append(RawItem(
            id=f"newsapi-{hashlib.sha1((url or title).encode()).hexdigest()[:16]}",
            source="news",
            text=f"{title}. {desc}" if desc else title,
            url=url, published_at=ts))
    return items


def fetch_newsapi(max_records: int = 25) -> list[RawItem]:
    if not config.NEWSAPI_KEY:
        record("newsapi", 0, "NEWSAPI_KEY is not set")
        return []
    params = {"q": config.NEWSAPI_QUERY, "language": "en", "sortBy": "publishedAt",
              "pageSize": str(max_records)}
    try:
        resp = http_get(config.NEWSAPI_URL, params=params, label="newsapi",
                        headers={"X-Api-Key": config.NEWSAPI_KEY}, timeout=15)
        payload = resp.json()
    except FetchError as exc:
        record("newsapi", 0, str(exc))
        return []
    except ValueError:
        record("newsapi", 0, "NewsAPI did not return JSON")
        return []
    if payload.get("status") == "error":
        record("newsapi", 0, f"{payload.get('code')}: {payload.get('message')}")
        return []
    items = parse_newsapi(payload)
    record("newsapi", len(items), None if items else "NewsAPI returned no articles")
    return items