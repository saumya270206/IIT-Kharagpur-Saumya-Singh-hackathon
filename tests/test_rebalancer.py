"""Module A: sentiment-driven tactical rebalancing."""

import pytest

from module_a import INDEX_TICKERS, IndexRebalancer, RebalanceConfig, equal_weights
from risk_engine.models import EventClass, RiskSignal
from risk_engine.store import SignalStore

_n = 0


def sig(ticker, sentiment, impact, event=EventClass.EARNINGS):
    global _n
    _n += 1
    return RiskSignal(signal_id=f"s{_n}", item_id=f"i{_n}", source="news", text="x",
                      ticker=ticker, sentiment=sentiment, event_class=event, impact=impact)


@pytest.fixture
def store(tmp_path):
    return SignalStore(tmp_path / "t.db")


@pytest.fixture
def reb(store):
    return IndexRebalancer(store)


def test_equal_weights_helper():
    w = equal_weights()
    assert set(w) == set(INDEX_TICKERS)
    assert sum(w.values()) == pytest.approx(1.0)


def test_starts_equal_weighted(reb):
    assert all(v == pytest.approx(1 / len(INDEX_TICKERS)) for v in reb.weights.values())


def test_positive_sentiment_increases_weight(reb):
    before = reb.weights["TSLA"]
    out = reb.apply(sig("TSLA", 0.9, 8))
    assert out["new_weight"] > before and reb.weights["TSLA"] > before


def test_negative_sentiment_decreases_weight(reb):
    before = reb.weights["TSLA"]
    reb.apply(sig("TSLA", -0.9, 8))
    assert reb.weights["TSLA"] < before


def test_impact_scales_the_move(store):
    a, b = IndexRebalancer(store), IndexRebalancer(SignalStore(store.db_path.with_name("b.db")))
    low = a.apply(sig("AAPL", 0.8, 2))["delta"]
    high = b.apply(sig("AAPL", 0.8, 9))["delta"]
    assert high > low > 0


def test_weights_sum_to_one_after_every_event(reb):
    for i, t in enumerate(INDEX_TICKERS * 2):
        reb.apply(sig(t, 1.0 if i % 2 else -1.0, 10))
        assert sum(reb.weights.values()) == pytest.approx(1.0)


def test_bounds_and_turnover_capped_under_extremes(reb):
    cfg = reb.cfg
    for i in range(60):
        out = reb.apply(sig("TSLA" if i % 3 else "AMD", 1.0 if i % 2 else -1.0, 10))
        assert out["turnover"] <= cfg.turnover_cap + 1e-9
        for w in reb.weights.values():
            assert cfg.min_weight - 1e-9 <= w <= cfg.max_weight + 1e-9


def test_custom_config_turnover_cap(store):
    reb = IndexRebalancer(store, RebalanceConfig(sensitivity=5.0, turnover_cap=0.01))
    out = reb.apply(sig("TSLA", 1.0, 10))
    assert out["turnover"] <= 0.01 + 1e-9


def test_out_of_index_and_unresolved_tickers_ignored(reb, store):
    assert reb.apply(sig("ORCL", 1.0, 9)) is None     # not in the mock index
    assert reb.apply(sig(None, -1.0, 9)) is None      # unresolved entity
    assert store.snapshot_times() == []


def test_snapshot_persisted_and_resumable(store):
    r1 = IndexRebalancer(store)
    r1.apply(sig("MSFT", 1.0, 9))
    r1.apply(sig("NFLX", -1.0, 9))
    assert len(store.snapshot_times()) == 2
    assert len(store.get_weight_history()) == 2 * len(INDEX_TICKERS)
    r2 = IndexRebalancer(store)   # a restart resumes from the stored snapshot
    for t in INDEX_TICKERS:
        assert r2.weights[t] == pytest.approx(r1.weights[t])
