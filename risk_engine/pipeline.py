"""The engine's orchestration layer: ingest -> NLP -> persist -> publish.

Downstream modules never get imported here. They *subscribe*:

    pipeline.subscribe(rebalancer.apply)      # Module A (sentiment)
    pipeline.subscribe(stress_tester.apply)   # Module B (event class + impact)

Every new RiskSignal is stored (SQLite + JSONL) and then pushed to each
subscriber. A failing subscriber is logged and isolated - it can never stop the
signal feed or starve the other modules.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Iterable, Optional

from risk_engine import config
from risk_engine.ingestion import (fetch_gdelt, fetch_newsapi, fetch_rss, fetch_yahoo,
                                    load_datasets)
from risk_engine.ingestion import http as ingest_http
from risk_engine.models import RawItem, RiskSignal
from risk_engine.nlp.analyzer import RiskAnalyzer
from risk_engine.store import SignalStore

log = logging.getLogger("risk_engine.pipeline")

Handler = Callable[[RiskSignal], Any]

# "live" = GDELT first, then Google News RSS, then Yahoo Finance RSS, then NewsAPI
# (only if NEWSAPI_KEY is set) - the first feed that returns headlines wins.
SOURCES: tuple[str, ...] = ("datasets", "gdelt", "rss", "yahoo", "newsapi", "live")

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _text_key(text: str) -> str:
    """Normalised fingerprint so the same story from two feeds is only scored once."""
    return _NORM_RE.sub(" ", text.lower()).strip()[:140]


class Pipeline:
    def __init__(
        self,
        store: SignalStore | None = None,
        analyzer: RiskAnalyzer | None = None,
    ) -> None:
        self.store = store or SignalStore()
        self.analyzer = analyzer or RiskAnalyzer()
        self.handlers: list[Handler] = []
        self._lock = threading.RLock()
        # fingerprints of already-seen non-manual items (survives restarts)
        self._seen: set[str] = {
            _text_key(s.text)
            for s in self.store.get_signals(limit=5000)
            if s.source != "manual"
        }

    # ------------------------------------------------------------ pub / sub
    def subscribe(self, handler: Handler) -> Handler:
        self.handlers.append(handler)
        return handler

    def _publish(self, signal: RiskSignal) -> None:
        for handler in self.handlers:
            try:
                handler(signal)
            except Exception:  # noqa: BLE001 - isolate subscriber failures
                log.exception("subscriber %r failed on signal %s",
                              getattr(handler, "__qualname__", handler),
                              signal.signal_id)

    # ----------------------------------------------------------- processing
    def process(self, item: RawItem) -> Optional[RiskSignal]:
        """Analyze one item. Returns the signal, or None for a duplicate."""
        with self._lock:
            key = _text_key(item.text)
            if self.store.has_item(item.id):
                return None
            if item.source != "manual" and key in self._seen:
                return None

            signal = self.analyzer.analyze(item)
            if not self.store.add_signal(signal):
                return None
            if item.source != "manual":
                self._seen.add(key)
            self._publish(signal)
            return signal

    def process_many(self, items: Iterable[RawItem]) -> list[RiskSignal]:
        out = []
        for item in sorted(items, key=lambda i: i.published_at):
            sig = self.process(item)
            if sig is not None:
                out.append(sig)
        return out

    # ------------------------------------------------------------ ingestion
    def fetch(self, source: str) -> list[RawItem]:
        if source == "datasets":
            return load_datasets()
        if source == "gdelt":
            return fetch_gdelt()
        if source == "rss":
            return fetch_rss()
        if source == "yahoo":
            return fetch_yahoo()
        if source == "newsapi":
            return fetch_newsapi()
        if source == "live":
            chain = [("gdelt", fetch_gdelt), ("rss", fetch_rss), ("yahoo", fetch_yahoo)]
            if config.NEWSAPI_KEY:
                chain.append(("newsapi", fetch_newsapi))
            for name, fn in chain:
                items = fn()
                if items:
                    log.info("live feed: using %s (%d headlines)", name, len(items))
                    self._last_live_feed = name
                    return items
            self._last_live_feed = None
            return []
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")

    def run_cycle(self, sources: Iterable[str] = ("datasets",)) -> dict:
        """One ingest cycle across the requested sources."""
        summary: dict[str, Any] = {"fetched": 0, "new_signals": 0,
                                   "duplicates": 0, "per_source": {}}
        for src in sources:
            items = self.fetch(src)
            new = self.process_many(items)
            info: dict[str, Any] = {"fetched": len(items), "new": len(new)}
            if src == "live" and items:
                info["feed"] = getattr(self, "_last_live_feed", None)
            if not items and src != "datasets":
                # surface *why* nothing came back instead of a silent zero
                feeds = ("gdelt", "rss", "yahoo", "newsapi") if src == "live" else (src,)
                info["errors"] = {f: ingest_http.STATUS[f]["error"] for f in feeds
                                  if f in ingest_http.STATUS and ingest_http.STATUS[f]["error"]}
                info["hint"] = "run `python run_engine.py diagnose` to test your connection"
            summary["per_source"][src] = info
            summary["fetched"] += len(items)
            summary["new_signals"] += len(new)
            summary["duplicates"] += len(items) - len(new)
        return summary