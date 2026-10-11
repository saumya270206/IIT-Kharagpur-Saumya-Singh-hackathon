"""GDELT is queried in several short batches (one long query is rejected by GDELT)."""

import pytest

from risk_engine import config
from risk_engine.ingestion import gdelt, http as ih
from risk_engine.models import RawItem


def item(i):
    return RawItem(id=f"gdelt-{i}", source="news", text=f"headline {i}", url=None,
                   published_at=gdelt._parse_seendate(None))


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    ih.STATUS.clear()


def test_default_queries_are_short_and_cover_index_members():
    assert len(config.GDELT_QUERIES) >= 2
    assert all(len(q) < 200 for q in config.GDELT_QUERIES)
    joined = " ".join(config.GDELT_QUERIES)
    for name in ("Apple", "Tesla", "Boeing"):
        assert name in joined
    assert "Nvidia" not in joined          # not an index member


def test_batches_are_merged_and_deduplicated(monkeypatch):
    results = iter([[item(1), item(2)], [item(2), item(3)], [item(4)]])
    monkeypatch.setattr(gdelt, "_fetch_one", lambda q, n, t: next(results))
    out = gdelt.fetch_gdelt()
    assert [i.id for i in out] == ["gdelt-1", "gdelt-2", "gdelt-3", "gdelt-4"]
    assert ih.STATUS["gdelt"]["ok"] and ih.STATUS["gdelt"]["count"] == 4


def test_rejected_batch_is_skipped_others_still_used(monkeypatch):
    calls = []

    def fake(q, n, t):
        calls.append(q)
        if len(calls) == 1:
            raise gdelt.FetchError("GDELT answered with text, not JSON: 'too short or too long'")
        return [item(len(calls))]

    monkeypatch.setattr(gdelt, "_fetch_one", fake)
    out = gdelt.fetch_gdelt()
    assert len(calls) == len(config.GDELT_QUERIES) and len(out) == len(calls) - 1
    assert ih.STATUS["gdelt"]["ok"]


def test_network_error_stops_early_and_is_reported(monkeypatch):
    calls = []

    def fake(q, n, t):
        calls.append(q)
        raise gdelt.FetchError("HTTP 429 rate limited")

    monkeypatch.setattr(gdelt, "_fetch_one", fake)
    assert gdelt.fetch_gdelt() == []
    assert len(calls) == 1
    assert "429" in ih.STATUS["gdelt"]["error"]


def test_custom_query_runs_once_with_full_record_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(gdelt, "_fetch_one", lambda q, n, t: seen.append((q, n)) or [item(1)])
    gdelt.fetch_gdelt(query="Apple stock", max_records=40)
    assert seen == [("Apple stock", 40)]