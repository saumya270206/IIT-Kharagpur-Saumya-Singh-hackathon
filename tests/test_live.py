"""Module A live panel: forward tracking after the Kaggle price window (no network)."""

import numpy as np
import pandas as pd
import pytest

from module_a.index import INDEX_TICKERS
from module_a.live import (fetch_live_prices, index_curve, live_cutoff, live_weights,
                           run_live_tracking)

T = INDEX_TICKERS
EQ = 1.0 / len(T)


def snapshot(ts, boost=None, amount=0.0):
    """Full weight snapshot at ``ts``; ``boost`` gets +amount, the rest share the cost."""
    w = {t: EQ for t in T}
    if boost:
        for t in T:
            w[t] += amount if t == boost else -amount / (len(T) - 1)
    return [{"ts": ts, "ticker": t, "weight": w[t]} for t in T]


def prices(days, daily_move=None):
    """Flat 100 prices, with optional per-ticker daily growth."""
    idx = pd.bdate_range(days[0], days[1])
    out = pd.DataFrame(100.0, index=idx, columns=T)
    for t, g in (daily_move or {}).items():
        out[t] = 100.0 * (1 + g) ** np.arange(len(idx))
    out.index.name = "Date"
    return out


def test_cutoff_is_midnight_after_last_price_day():
    assert live_cutoff(pd.Timestamp("2022-09-29")) == pd.Timestamp("2022-09-30", tz="UTC")


def test_no_live_snapshots_returns_none():
    rows = snapshot("2022-03-01T10:00:00+00:00", "AAPL", 0.02)
    assert live_weights(rows, T, live_cutoff(pd.Timestamp("2022-09-29"))) is None


def test_live_weights_compare_against_live_start():
    old = snapshot("2022-03-01T10:00:00+00:00", "AAPL", 0.02)          # historical tilt
    new = snapshot("2026-10-06T11:00:00+00:00", "TSLA", 0.03)          # live rebalance
    lw = live_weights(old + new, T, live_cutoff(pd.Timestamp("2022-09-29")))
    assert lw.n_rebalances == 1
    assert lw.table.loc["AAPL", "at_live_start"] == pytest.approx(EQ + 0.02)
    assert lw.table.loc["TSLA", "current"] == pytest.approx(EQ + 0.03)
    assert lw.table["change"].sum() == pytest.approx(0.0, abs=1e-12)
    assert lw.active_tilt > 0 and lw.live_turnover > 0


def test_tracking_has_no_lookahead_and_reflects_live_weights():
    # live snapshot Tue 6 Oct 11:00 tilts to TSLA; TSLA grows 1%/day
    rows = snapshot("2026-10-06T11:00:00+00:00", "TSLA", 0.05)
    px = prices(("2026-10-01", "2026-10-09"), {"TSLA": 0.01})
    first = pd.Timestamp("2026-10-06T11:00:00", tz="UTC")
    daily, base = run_live_tracking(rows, T, px, first)
    assert base == pd.Timestamp("2026-10-06")
    assert list(daily.index) == list(pd.bdate_range("2026-10-07", "2026-10-09"))
    # base date return is excluded (snapshot not yet public on that day) and the first
    # tracked day already uses the new tilt -> sentiment beats equal weight
    assert (daily["active"] > 0).all()
    assert daily["equal_weight"].iloc[0] == pytest.approx(EQ * 0.01)


def test_tracking_empty_until_a_full_trading_day_passes():
    rows = snapshot("2026-10-09T11:00:00+00:00", "TSLA", 0.05)
    px = prices(("2026-10-05", "2026-10-09"))
    daily, base = run_live_tracking(rows, T, px, pd.Timestamp("2026-10-09T11:00", tz="UTC"))
    assert daily.empty and base == pd.Timestamp("2026-10-09")


def test_weekend_snapshot_uses_friday_close_as_base():
    rows = snapshot("2026-10-10T09:00:00+00:00", "AAPL", 0.04)         # Saturday
    px = prices(("2026-10-05", "2026-10-13"), {"AAPL": 0.02})
    daily, base = run_live_tracking(rows, T, px, pd.Timestamp("2026-10-10T09:00", tz="UTC"))
    assert base == pd.Timestamp("2026-10-09")
    assert daily.index[0] == pd.Timestamp("2026-10-12")                # Monday uses tilt
    assert daily["active"].iloc[0] > 0


def test_empty_prices_are_handled():
    daily, base = run_live_tracking(snapshot("2026-10-06T11:00:00+00:00"), T,
                                    pd.DataFrame(), pd.Timestamp("2026-10-06", tz="UTC"))
    assert daily.empty and base is None


def test_index_curve_starts_at_100():
    rows = snapshot("2026-10-06T11:00:00+00:00", "TSLA", 0.05)
    px = prices(("2026-10-05", "2026-10-09"), {"TSLA": 0.01})
    daily, base = run_live_tracking(rows, T, px, pd.Timestamp("2026-10-06T11:00", tz="UTC"))
    curve = index_curve(daily, base)
    assert curve.iloc[0].tolist() == [100.0, 100.0]
    assert len(curve) == len(daily) + 1


def test_fetch_failure_returns_reason_not_exception(monkeypatch):
    import sys, types
    fake = types.SimpleNamespace(download=lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    df, err = fetch_live_prices(T, "2026-10-01")
    assert df.empty and "offline" in err