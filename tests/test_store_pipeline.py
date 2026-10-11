"""Store + pipeline: persistence, dedup, publish/subscribe, subscriber isolation."""

import json

import pytest

from risk_engine import config
from risk_engine.models import RawItem
from risk_engine.nlp.analyzer import RiskAnalyzer
from risk_engine.pipeline import SOURCES, Pipeline
from risk_engine.store import SignalStore


@pytest.fixture
def store(tmp_path):
    return SignalStore(tmp_path / "r.db")


@pytest.fixture
def pipe(store):
    return Pipeline(store, RiskAnalyzer(backend="off"))


def item(i, text, source="news", **kw):
    return RawItem(id=str(i), source=source, text=text, **kw)


def test_signal_roundtrip_and_jsonl_mirror(pipe, store):
    sig = pipe.process(item(1, "Tesla recalls 2 million vehicles over defect"))
    assert sig.ticker == "TSLA"
    got = store.get_signals(limit=5)
    assert len(got) == 1 and got[0].signal_id == sig.signal_id
    lines = store.jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["signal_id"] == sig.signal_id


def test_duplicates_by_id_and_by_text(pipe):
    assert pipe.process(item(1, "Nvidia hits record revenue")) is not None
    assert pipe.process(item(1, "other words entirely")) is None            # same id
    assert pipe.process(item(2, "NVIDIA hits record  revenue!")) is None   # same story, new id


def test_manual_items_are_never_text_deduped(pipe):
    assert pipe.process(item("m1", "Apple surges", source="manual")) is not None
    assert pipe.process(item("m2", "Apple surges", source="manual")) is not None


def test_dedup_survives_restart(store):
    p1 = Pipeline(store, RiskAnalyzer(backend="off"))
    p1.process(item(1, "Walmart raises guidance"))
    p2 = Pipeline(store, RiskAnalyzer(backend="off"))
    assert p2.process(item(99, "Walmart raises guidance")) is None


def test_subscribers_receive_signals_and_failures_are_isolated(pipe):
    seen = []

    def bad(sig):
        raise RuntimeError("boom")

    pipe.subscribe(bad)
    pipe.subscribe(seen.append)
    sig = pipe.process(item(1, "Microsoft to acquire AI startup in all-cash deal"))
    assert sig is not None and seen == [sig]


def test_filters(pipe, store):
    pipe.process(item(1, "$AAPL surges on record iPhone demand"))
    pipe.process(item(2, "$TSLA plunges on recall", source="twitter"))
    assert [s.ticker for s in store.get_signals(ticker="TSLA")] == ["TSLA"]
    assert [s.source for s in store.get_signals(source="twitter")] == ["twitter"]
    assert store.count_signals() == 2
    assert store.get_signals(limit=1)[0].ticker == "TSLA"   # newest first


def test_run_cycle_on_kaggle_datasets(pipe, monkeypatch):
    monkeypatch.setattr(config, "DATASET_TWEET_LIMIT", 150)
    monkeypatch.setattr(config, "DATASET_NEWS_LIMIT", 50)
    first = pipe.run_cycle(["datasets"])
    assert first["new_signals"] > 20 and first["new_signals"] <= first["fetched"]
    again = pipe.run_cycle(["datasets"])
    assert again["new_signals"] == 0 and again["duplicates"] == again["fetched"]


def test_unknown_source_rejected(pipe):
    assert "live" in SOURCES
    with pytest.raises(ValueError):
        pipe.fetch("telegram")


def test_reset_clears_everything(pipe, store):
    pipe.process(item(1, "Amazon plunges after outage"))
    store.reset()
    assert store.count_signals() == 0 and store.jsonl_path.read_text() == ""
