"""Module A dashboard: live view of sentiment-driven index weights.

Run with:  python run_dashboard.py   (or: streamlit run dashboard/app.py)
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from module_a.index import INDEX_NAME, INDEX_TICKERS, equal_weights
from module_a.live import (fetch_live_prices, index_curve, live_cutoff,
                           live_weights, run_live_tracking)
from risk_engine.config import API_HOST, API_PORT
from risk_engine.store import SignalStore

st.set_page_config(page_title="Tactical Index Rebalancer", layout="wide")
API = f"http://{API_HOST}:{API_PORT}"
EQ = 1.0 / len(INDEX_TICKERS)
EVENT_COLORS = px.colors.qualitative.Set2


# --------------------------------------------------------------------- helpers
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API}/health", timeout=1)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def trigger_cycle(sources: list[str]) -> dict:
    try:
        r = requests.post(f"{API}/pipeline/run", json={"sources": sources}, timeout=180)
        return r.json() if r.ok else {"error": f"API returned {r.status_code}"}
    except requests.RequestException as exc:
        return {"error": f"API unreachable: {exc}"}


def load_weights_df(store: SignalStore) -> pd.DataFrame:
    rows = store.get_weight_history()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    return df.sort_values("ts")


def load_signals_df(store: SignalStore, limit: int = 200) -> pd.DataFrame:
    sigs = store.get_signals(limit=limit)
    if not sigs:
        return pd.DataFrame()
    df = pd.DataFrame([s.to_json() for s in sigs])
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True)
    df["event_at"] = pd.to_datetime(df["event_at"], format="ISO8601", utc=True)
    return df


@st.cache_data(ttl=900, show_spinner="Fetching live prices...")
def cached_live_prices(tickers: tuple[str, ...], start: str):
    """Yahoo Finance closes, cached 15 min so the 10 s auto-refresh never hits the network."""
    return fetch_live_prices(list(tickers), start)


# --------------------------------------------------------------------- sidebar
st.sidebar.title("⚖️ Tactical Index Rebalancer")
st.sidebar.caption(f"{INDEX_NAME} · {len(INDEX_TICKERS)} S&P 100 constituents")

health = api_health()
if health:
    st.sidebar.success(
        f"Engine online · {health['signals']} signals · "
        f"backend: {health['nlp_backend']}"
    )
else:
    st.sidebar.error("Engine API offline — start it with `python run_api.py`")

st.sidebar.subheader("Ingest")
if st.sidebar.button("📰 Replay Kaggle datasets", width="stretch"):
    result = trigger_cycle(["datasets"])
    st.sidebar.write(result if "error" in result else f"✅ {result}")
if st.sidebar.button("🌐 Ingest live news (GDELT/RSS)", width="stretch"):
    with st.sidebar:
        with st.spinner("Fetching headlines..."):
            result = trigger_cycle(["live"])
    st.sidebar.write(result if "error" in result else f"✅ {result}")

st.sidebar.subheader("Filter")
selected = st.sidebar.multiselect(
    "Tickers", INDEX_TICKERS, default=INDEX_TICKERS,
    help="Which index members to plot",
)

# ----------------------------------------------------------------------- page
st.title("📈 AI Sentiment-Driven Index Weights")
st.caption(
    "Module A subscribes to Risk Signals from the AI/NLP Risk Engine: "
    "positive sentiment raises a stock's weight, negative sentiment lowers it, "
    "scaled by the event's impact score."
)

store = SignalStore()
wdf = load_weights_df(store)
sdf = load_signals_df(store)

# snapshot metrics (no history yet -> friendly empty state)
if wdf.empty:
    st.info("No rebalances yet. Click **Replay Kaggle datasets** in the sidebar "
            "to run the engine over the tweets + news datasets.")
    st.stop()

snap_times = sorted(wdf["ts"].unique())
latest_ts = snap_times[-1]
latest = wdf[wdf["ts"] == latest_ts].set_index("ticker")["weight"]
prev = wdf[wdf["ts"] == snap_times[-2]].set_index("ticker")["weight"] if len(snap_times) > 1 else latest
latest_sig = sdf.iloc[0] if not sdf.empty else None

turnover = float((latest - prev).abs().sum() / 2)
top = latest.sort_values(ascending=False)
over_t, over_w = top.index[0], float(top.iloc[0])
under_t, under_w = top.index[-1], float(top.iloc[-1])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Signals processed", len(sdf) if not sdf.empty else store.count_signals())
m2.metric("Rebalance events", len(snap_times))
m3.metric("Largest weight", f"{over_t} {over_w:.2%}", f"{over_w - EQ:+.2%} vs equal")
m4.metric("Smallest weight", f"{under_t} {under_w:.2%}", f"{under_w - EQ:+.2%} vs equal")
m5.metric("Turnover last event", f"{turnover:.2%}")


@st.fragment(run_every=timedelta(seconds=10))
def live_view() -> None:
    """Re-renders the live data every 10 seconds."""
    store = SignalStore()
    wdf = load_weights_df(store)
    sdf = load_signals_df(store)
    if wdf.empty:
        return

    fig = px.line(
        wdf[wdf["ticker"].isin(selected)],
        x="ts", y="weight", color="ticker",
        markers=True,
        title="Index weights over time (driven by incoming risk signals)",
        labels={"weight": "Weight", "ts": "Time (UTC)", "ticker": "Stock"},
    )
    fig.update_yaxes(tickformat=".1%")
    fig.add_hline(y=EQ, line_dash="dash", line_color="gray",
                  annotation_text="equal weight")
    st.plotly_chart(fig, width="stretch")

    latest_ts = wdf["ts"].max()
    latest = wdf[wdf["ts"] == latest_ts].set_index("ticker")["weight"]
    snap = latest.sort_values(ascending=False)
    bar = go.Figure(go.Bar(
        x=snap.index, y=snap.values,
        marker_color=["#2ca02c" if v >= EQ else "#d62728" for v in snap.values],
        text=[f"{v:.2%}" for v in snap.values], textposition="outside",
    ))
    bar.add_hline(y=EQ, line_dash="dash", line_color="gray",
                  annotation_text="equal weight")
    bar.update_layout(
        title="Current weights (green = at/above equal, red = below)",
        yaxis_title="Weight", yaxis_tickformat=".1%", height=380,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(bar, width="stretch")
    with right:
        if not sdf.empty:
            sc = px.scatter(
                sdf, x="sentiment", y="impact", color="event_class",
                size="confidence", hover_data={"text": True, "ticker": True},
                title="Latest risk signals: sentiment vs. impact",
                color_discrete_sequence=EVENT_COLORS,
                range_x=[-1.05, 1.05], range_y=[0, 10.5],
            )
            sc.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(sc, width="stretch")

    st.subheader("Signal feed")
    if not sdf.empty:
        show = sdf.head(40)[
            ["created_at", "source", "ticker", "event_class",
             "sentiment", "impact", "confidence", "text"]
        ].rename(columns={"created_at": "time", "event_class": "event"})
        show["text"] = show["text"].str.slice(0, 90)
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.caption("No signals yet.")


st.subheader("Historical back-test on real prices (Kaggle yfinance data)")
weight_rows = store.get_weight_history()
hist_end = None
try:
    from module_a.backtest import run_backtest, summarize
    daily = run_backtest(weight_rows, INDEX_TICKERS)
    hist_end = daily.index.max()
    stats = summarize(daily)
    curve = (1 + daily[["sentiment_index", "equal_weight"]]).cumprod() * 100
    bt = px.line(curve, title="Sentiment-driven index vs equal-weight (rebased to 100)",
                 labels={"value": "Index level", "Date": "Date", "variable": ""})
    st.plotly_chart(bt, width="stretch")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sentiment index return", f"{stats['sentiment_index']['total_return']:+.2%}")
    b2.metric("Equal-weight return", f"{stats['equal_weight']['total_return']:+.2%}")
    b3.metric("Excess return", f"{stats['excess_return']:+.2%}")
    b4.metric("Tracking error (ann.)", f"{stats['tracking_error']:.2%}")
    st.caption("Weights applied to day t are the last snapshot made before day t "
               "(no look-ahead). Prices: data/raw/stock_yfinance_data.csv. "
               f"This replay covers {daily.index.min():%d %b %Y} - {daily.index.max():%d %b %Y} "
               "only, so live-news rebalances do not change it - they are tracked in the "
               "live panel below.")
except FileNotFoundError as exc:
    st.info(str(exc))


# ------------------------------------------------------------- live index panel
def render_live_index(rows: list[dict], cutoff: pd.Timestamp) -> None:
    head, refresh = st.columns([5, 1])
    head.subheader("Live index: forward tracking from live news")
    if refresh.button("Refresh prices", width="stretch"):
        cached_live_prices.clear()
        st.rerun()

    lw = live_weights(rows, INDEX_TICKERS, cutoff)
    if lw is None:
        st.info("No live-news rebalances yet. Click **Ingest live news (GDELT/RSS)** in the "
                "sidebar. Headlines about index members will move the weights, and this "
                "panel then tracks them against fresh market prices.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live rebalances", lw.n_rebalances)
    m2.metric("First live rebalance", f"{lw.first_ts:%d %b %Y}")
    m3.metric("Active tilt vs equal", f"{lw.active_tilt:.2%}")
    m4.metric("Turnover since live start", f"{lw.live_turnover:.2%}")

    # --- weights: how live news moved the book
    left, right = st.columns(2)
    chg = lw.table["change"].sort_values()
    bar = go.Figure(go.Bar(
        x=chg.index, y=chg.values * 100,
        marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in chg.values],
        text=[f"{v * 100:+.2f}" for v in chg.values], textposition="outside",
    ))
    bar.update_layout(title="Weight change since live period began (pp)",
                      yaxis_title="Percentage points", height=360)
    left.plotly_chart(bar, width="stretch")
    pct = st.column_config.NumberColumn
    right.dataframe(
        (lw.table * 100).sort_values("current", ascending=False),
        width="stretch",
        column_config={
            "equal": pct("Equal", format="%.2f%%"),
            "at_live_start": pct("At live start", format="%.2f%%"),
            "current": pct("Current", format="%.2f%%"),
            "change": pct("Change (pp)", format="%+.2f"),
        },
    )

    # --- performance: live weights vs fresh prices
    prices, err = cached_live_prices(
        tuple(INDEX_TICKERS), (lw.first_ts - pd.Timedelta(days=7)).date().isoformat())
    if err:
        cached_live_prices.clear()   # do not cache a failure for 15 minutes
        st.warning(f"Live prices unavailable - {err}. Weights above are still current.")
        return
    ldaily, base = run_live_tracking(rows, INDEX_TICKERS, prices, lw.first_ts)
    if ldaily.empty:
        st.info("Forward tracking starts after the first full trading day following the "
                f"first live rebalance ({lw.first_ts:%d %b %Y %H:%M} UTC). "
                "Check back after the next market close.")
        return

    lc = index_curve(ldaily, base)
    fig = px.line(lc, title=f"Live: sentiment-driven index vs equal-weight "
                            f"(100 = close of {base:%d %b %Y})",
                  labels={"value": "Index level", "variable": ""})
    st.plotly_chart(fig, width="stretch")

    sent = float((1 + ldaily["sentiment_index"]).prod() - 1)
    eqw = float((1 + ldaily["equal_weight"]).prod() - 1)
    days = len(ldaily)
    te = (f"{float(ldaily['active'].std() * np.sqrt(252)):.2%}" if days >= 5
          else "n/a (<5 days)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live sentiment return", f"{sent:+.2%}")
    c2.metric("Live equal-weight return", f"{eqw:+.2%}")
    c3.metric("Live excess return", f"{sent - eqw:+.2%}")
    c4.metric("Tracking error (ann.)", te, f"{days} trading day(s)", delta_color="off")
    st.caption("Prices: Yahoo Finance via yfinance (cached 15 min; the latest bar can be "
               "intraday). Weights applied to day t are the last snapshot made before day t "
               "(no look-ahead). The sentiment index starts from the weights the engine held "
               "when live news began (not exactly equal weight), so the lines can differ from "
               "day one. Only the 15 index members move weights; headlines on other "
               "companies are ignored.")


if hist_end is not None:
    render_live_index(weight_rows, live_cutoff(hist_end))

live_view()

st.caption(f"Auto-refreshes every 10s · data source: SQLite ({store.db_path.name}) · "
           f"signals also mirrored to signals.jsonl and served by the API on port {API_PORT}.")