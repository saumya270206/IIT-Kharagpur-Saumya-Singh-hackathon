"""Live-ingestion code path, exercised against a local server imitating the real feeds."""

import pytest

from risk_engine import config
from risk_engine.ingestion import http as ih
from risk_engine.ingestion import fetch_gdelt, fetch_newsapi, fetch_rss, fetch_yahoo
from risk_engine.ingestion.rss import clean_title
from risk_engine.pipeline import Pipeline
from risk_engine.store import SignalStore
from tests import mock_feeds


@pytest.fixture()
def feeds(monkeypatch):
    srv, base = mock_feeds.start()
    monkeypatch.setattr(config, "GDELT_URL", f"{base}/gdelt")
    monkeypatch.setattr(config, "GOOGLE_NEWS_URL", f"{base}/gnews")
    monkeypatch.setattr(config, "YAHOO_RSS_URL", f"{base}/yahoo")
    monkeypatch.setattr(config, "NEWSAPI_URL", f"{base}/newsapi")
    monkeypatch.setattr(config, "YAHOO_TICKERS", ["AAPL", "XOM", "MSFT"])
    monkeypatch.setattr(ih.time, "sleep", lambda s: None)     # no real back-off waits
    ih.STATUS.clear()
    yield mock_feeds.STATE
    srv.shutdown()


def test_gdelt_parses_articles(feeds):
    items = fetch_gdelt()
    assert len(items) == 6 and items[0].text.startswith("Nvidia")
    assert ih.STATUS["gdelt"]["ok"]
    assert len({i.id for i in items}) == 6             # stable, unique ids


def test_gdelt_retries_after_429(feeds):
    feeds["gdelt"] = "429_once"
    assert len(fetch_gdelt()) == 6


def test_gdelt_text_error_is_reported_not_crashed(feeds):
    feeds["gdelt"] = "text_error"
    assert fetch_gdelt() == []
    assert "not JSON" in ih.STATUS["gdelt"]["error"]
    assert "too short" in ih.STATUS["gdelt"]["error"]


def test_gdelt_server_error_reported(feeds):
    feeds["gdelt"] = "down"
    assert fetch_gdelt() == []
    assert "500" in ih.STATUS["gdelt"]["error"]


def test_google_news_rss_strips_publisher_suffix(feeds):
    items = fetch_rss()
    assert len(items) == 4
    assert items[0].text == "Apple unveils new iPhone with record preorders"
    assert all(" - Reuters" not in i.text and " - CNBC" not in i.text for i in items)


def test_yahoo_sets_ticker_hint(feeds):
    items = fetch_yahoo()
    assert {i.ticker for i in items} == {"AAPL", "XOM"}


def test_newsapi_needs_key_then_works(feeds, monkeypatch):
    assert fetch_newsapi() == [] and "NEWSAPI_KEY" in ih.STATUS["newsapi"]["error"]
    monkeypatch.setattr(config, "NEWSAPI_KEY", "k")
    assert fetch_newsapi()[0].text.startswith("AMD plunges")


def test_unreachable_host_gives_readable_error(monkeypatch):
    monkeypatch.setattr(config, "GDELT_URL", "http://127.0.0.1:9/gdelt")  # closed port
    monkeypatch.setattr(config, "HTTP_RETRIES", 0)
    ih.STATUS.clear()
    assert fetch_gdelt() == []
    assert ih.STATUS["gdelt"]["error"]


def test_live_chain_falls_back_gdelt_to_rss_to_yahoo(feeds, tmp_path):
    p = Pipeline(SignalStore(tmp_path / "l.db"))
    feeds["gdelt"] = "down"
    out = p.run_cycle(["live"])
    assert out["per_source"]["live"]["feed"] == "rss" and out["new_signals"] == 4
    feeds["gnews"] = "down"
    p2 = Pipeline(SignalStore(tmp_path / "l2.db"))
    out = p2.run_cycle(["live"])
    assert out["per_source"]["live"]["feed"] == "yahoo"


def test_all_feeds_down_reports_reasons(feeds, tmp_path):
    for k in ("gdelt", "gnews", "yahoo"):
        feeds[k] = "down"
    out = Pipeline(SignalStore(tmp_path / "d.db")).run_cycle(["live"])
    info = out["per_source"]["live"]
    assert out["new_signals"] == 0 and set(info["errors"]) == {"gdelt", "rss", "yahoo"}
    assert "diagnose" in info["hint"]


def test_live_end_to_end_produces_structured_signals(feeds, tmp_path):
    p = Pipeline(SignalStore(tmp_path / "e.db"))
    out = p.run_cycle(["live"])
    assert out["new_signals"] >= 5
    sigs = p.store.get_signals(limit=50)
    by = {s.ticker: s for s in sigs if s.ticker}
    assert by["NVDA"].sentiment > 0 and by["JPM"].sentiment < 0
    assert by["JPM"].event_class.value == "Credit Event"
    # re-running the same feed produces no duplicates
    assert p.run_cycle(["live"])["new_signals"] == 0


@pytest.mark.parametrize("raw,pub,expected", [
    ("Tesla recalls cars over defect - Reuters", None, "Tesla recalls cars over defect"),
    ("Tesla recalls cars over defect - Reuters", "Reuters", "Tesla recalls cars over defect"),
    ("Short - AP", None, "Short - AP"),                      # too short to strip safely
    ("No suffix here at all", None, "No suffix here at all"),
])
def test_clean_title(raw, pub, expected):
    assert clean_title(raw, pub) == expected