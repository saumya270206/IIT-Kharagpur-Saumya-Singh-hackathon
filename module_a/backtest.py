"""Module A back-test on real prices (Kaggle ``stock_yfinance_data.csv``).

The sentiment-tilted weights the rebalancer produced are replayed against the
daily closes of the same period and compared with a static equal-weight index.

No look-ahead: the weights applied to day *t*'s return (close[t]/close[t-1]) are the
last snapshot made strictly before day *t* (UTC date), i.e. only from tweets that
were already public.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine import config


def load_prices(path: Path | str | None = None) -> pd.DataFrame:
    """Wide DataFrame of adjusted closes: index = date, columns = ticker."""
    p = Path(path or config.PRICES_CSV)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - put stock_yfinance_data.csv in {config.RAW_DIR}")
    df = pd.read_csv(p, parse_dates=["Date"])
    px = df.pivot_table(index="Date", columns="Stock Name", values="Adj Close")
    return px.sort_index()


def weights_by_day(weight_rows: list[dict], dates: pd.DatetimeIndex,
                   tickers: list[str]) -> pd.DataFrame:
    """Last weight snapshot strictly before each trading day (equal weight before the first)."""
    eq = 1.0 / len(tickers)
    if not weight_rows:
        return pd.DataFrame(eq, index=dates, columns=tickers)
    w = pd.DataFrame(weight_rows)
    w["ts"] = pd.to_datetime(w["ts"], format="ISO8601", utc=True)
    snap = w.pivot_table(index="ts", columns="ticker", values="weight").sort_index()
    snap = snap.reindex(columns=tickers)
    day_start = pd.DatetimeIndex(dates).tz_localize("UTC")
    # position of the last snapshot with ts < start of the day
    pos = snap.index.searchsorted(day_start, side="left") - 1
    out = np.full((len(dates), len(tickers)), eq)
    ok = pos >= 0
    out[ok] = snap.to_numpy()[pos[ok]]
    return pd.DataFrame(out, index=dates, columns=tickers)


def run_backtest(weight_rows: list[dict], tickers: list[str],
                 prices: pd.DataFrame | None = None) -> pd.DataFrame:
    px = (prices if prices is not None else load_prices())
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers].dropna(how="all").ffill()
    ret = px.pct_change().iloc[1:]
    w = weights_by_day(weight_rows, ret.index, tickers)
    w = w.div(w.sum(axis=1), axis=0)
    eq = pd.Series(1.0 / len(tickers), index=tickers)
    out = pd.DataFrame({
        "sentiment_index": (w * ret).sum(axis=1),
        "equal_weight": (ret * eq).sum(axis=1),
    })
    out["active"] = out["sentiment_index"] - out["equal_weight"]
    return out


def summarize(daily: pd.DataFrame) -> dict:
    def stats(r: pd.Series) -> dict:
        cum = float((1 + r).prod() - 1)
        vol = float(r.std() * np.sqrt(252))
        ann = float((1 + cum) ** (252 / max(len(r), 1)) - 1)
        eq_curve = (1 + r).cumprod()
        mdd = float((eq_curve / eq_curve.cummax() - 1).min())
        return {"total_return": cum, "ann_vol": vol,
                "sharpe": ann / vol if vol else 0.0, "max_drawdown": mdd}
    te = float(daily["active"].std() * np.sqrt(252))
    return {"sentiment_index": stats(daily["sentiment_index"]),
            "equal_weight": stats(daily["equal_weight"]),
            "excess_return": float((1 + daily["sentiment_index"]).prod()
                                   - (1 + daily["equal_weight"]).prod()),
            "tracking_error": te, "days": int(len(daily))}
