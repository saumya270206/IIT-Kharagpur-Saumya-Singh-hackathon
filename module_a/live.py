"""Module A live panel: forward-tracking of the sentiment index after the Kaggle window.

The historical back-test (``module_a.backtest``) is pinned to the Kaggle price
file (Sep 2021 - Sep 2022), so weight snapshots created from *live* news (dated
today) can never meet a price day there. This module closes that gap:

  * ``live_cutoff``      - first instant that counts as "live" (day after the last Kaggle price)
  * ``live_weights``     - current weights vs. equal weight and vs. the start of the live period
  * ``fetch_live_prices``- fresh daily closes from Yahoo Finance (yfinance), failure-tolerant
  * ``run_live_tracking``- replays live weight snapshots against fresh prices, no look-ahead
  * ``index_curve``      - daily returns -> index levels rebased to 100

The tracking reuses ``run_backtest`` so the weighting rule is identical: the
weights applied to day *t* are the last snapshot made strictly before day *t* (UTC).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from module_a.backtest import run_backtest


# --------------------------------------------------------------------- helpers
def live_cutoff(last_price_day: pd.Timestamp | datetime) -> pd.Timestamp:
    """UTC midnight of the day after the last historical price day."""
    ts = pd.Timestamp(last_price_day)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize().tz_localize("UTC") + pd.Timedelta(days=1)


def _snapshots(weight_rows: list[dict], tickers: list[str]) -> pd.DataFrame:
    """Wide table: index = snapshot ts (UTC), columns = ticker, values = weight."""
    if not weight_rows:
        return pd.DataFrame(columns=tickers, dtype=float)
    w = pd.DataFrame(weight_rows)
    w["ts"] = pd.to_datetime(w["ts"], format="ISO8601", utc=True)
    snap = w.pivot_table(index="ts", columns="ticker", values="weight").sort_index()
    return snap.reindex(columns=tickers)


# ------------------------------------------------------------- live weights
@dataclass
class LiveWeights:
    table: pd.DataFrame        # index=ticker; equal, at_live_start, current, change
    n_rebalances: int          # snapshots made since the cutoff
    first_ts: pd.Timestamp     # first live snapshot (UTC)
    last_ts: pd.Timestamp      # latest live snapshot (UTC)

    @property
    def active_tilt(self) -> float:
        """One-way distance of the current weights from equal weight."""
        return float((self.table["current"] - self.table["equal"]).abs().sum() / 2)

    @property
    def live_turnover(self) -> float:
        """One-way turnover accumulated since the live period started."""
        return float(self.table["change"].abs().sum() / 2)


def live_weights(weight_rows: list[dict], tickers: list[str],
                 cutoff: pd.Timestamp) -> LiveWeights | None:
    """Summarise live rebalances; None when no snapshot exists at/after ``cutoff``."""
    snap = _snapshots(weight_rows, tickers)
    live = snap[snap.index >= cutoff]
    if live.empty:
        return None
    eq = 1.0 / len(tickers)
    before = snap[snap.index < cutoff]
    start = before.iloc[-1] if len(before) else pd.Series(eq, index=tickers)
    current = snap.iloc[-1]
    table = pd.DataFrame({
        "equal": eq,
        "at_live_start": start.astype(float),
        "current": current.astype(float),
    })
    table["change"] = table["current"] - table["at_live_start"]
    table.index.name = "ticker"
    return LiveWeights(table, int(len(live)), live.index[0], live.index[-1])


# --------------------------------------------------------------- live prices
def fetch_live_prices(tickers: list[str], start: pd.Timestamp | datetime | str
                      ) -> tuple[pd.DataFrame, str | None]:
    """Daily adjusted closes from Yahoo Finance. Returns (prices, error).

    Never raises: on any failure (no yfinance, offline, rate-limited) it returns
    an empty frame and a human-readable reason so the dashboard keeps working.
    """
    empty = pd.DataFrame()
    try:
        import yfinance as yf
    except ImportError:
        return empty, "yfinance is not installed (run: pip install yfinance)"
    try:
        end = datetime.now(timezone.utc).date() + timedelta(days=1)
        raw = yf.download(list(tickers), start=pd.Timestamp(start).date().isoformat(),
                          end=end.isoformat(), progress=False, auto_adjust=True,
                          threads=False)
    except Exception as exc:  # noqa: BLE001 - network/library errors must not crash the UI
        return empty, f"price download failed: {exc}"
    if raw is None or raw.empty or "Close" not in raw:
        return empty, "Yahoo Finance returned no prices"
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all")
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index = pd.DatetimeIndex(close.index).normalize()
    close.index.name = "Date"
    return close.sort_index(), None


# ----------------------------------------------------------- forward tracking
def run_live_tracking(weight_rows: list[dict], tickers: list[str],
                      live_prices: pd.DataFrame, first_live_ts: pd.Timestamp
                      ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Sentiment index vs equal weight from the first live rebalance onwards.

    Both start at 100 on the last close on/before the day of the first live
    snapshot (the *base date*). Returns (daily returns, base_date); the frame is
    empty until at least one full trading day has passed after the base date.
    """
    if live_prices is None or live_prices.empty:
        return pd.DataFrame(), None
    px = live_prices.copy()
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    d0 = first_live_ts.tz_convert("UTC").normalize().tz_localize(None)
    candidates = px.index[px.index <= d0]
    if len(candidates) == 0:
        return pd.DataFrame(), None
    base = candidates[-1]
    px = px.loc[base:]
    if len(px) < 2:
        return pd.DataFrame(), base
    return run_backtest(weight_rows, tickers, prices=px), base


def index_curve(daily: pd.DataFrame, base_date: pd.Timestamp) -> pd.DataFrame:
    """Daily returns -> index levels, with the base date shown as 100."""
    cols = ["sentiment_index", "equal_weight"]
    curve = (1 + daily[cols]).cumprod() * 100
    start = pd.DataFrame({c: [100.0] for c in cols}, index=pd.DatetimeIndex([base_date]))
    out = pd.concat([start, curve])
    out.index.name = "Date"
    return out