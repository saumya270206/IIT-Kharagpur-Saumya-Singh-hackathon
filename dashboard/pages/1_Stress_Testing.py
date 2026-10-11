"""Module B dashboard: event-driven portfolio stress testing.

Shows the wholesale book before/after each stress test triggered by the NLP
engine (event class + impact), plus an interactive what-if runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import STATUS_COLOR, api_get, api_post, money
from module_b.portfolio import load_portfolio
from module_b.scenarios import TEMPLATES, build_scenario
from module_b.store import StressStore
from module_b.stress import StressConfig, StressEngine
from risk_engine.models import EventClass
from risk_engine.store import SignalStore

st.set_page_config(page_title="Portfolio Stress Testing", layout="wide")

store = SignalStore()
results = StressStore(store)
portfolio = load_portfolio()
cfg = StressConfig()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ Stress Testing")
st.sidebar.caption("Module B · event-driven, wholesale banking book")
health = api_get("/health")
if health:
    st.sidebar.success(f"Engine online · {health['signals']} signals")
else:
    st.sidebar.error("Engine API offline — start `python run_api.py`")

conf = api_get("/stress/config")
rule = conf["rule"] if conf else (
    cfg.rule_text() if hasattr(cfg, "rule_text") else
    f"auto stress test when impact > {cfg.impact_threshold} and sentiment <= {cfg.max_sentiment}")
st.sidebar.info(f"**Trigger rule**\n\n{rule}")

if st.sidebar.button("📰 Replay Kaggle datasets", width="stretch"):
    out = api_post("/pipeline/run", {"sources": ["datasets"]})
    st.sidebar.write(out if "error" in out else f"✅ {out['new_signals']} new signals")
if st.sidebar.button("🌐 Ingest live news", width="stretch"):
    with st.spinner("Fetching headlines..."):
        out = api_post("/pipeline/run", {"sources": ["live"]})
    st.sidebar.write(out if "error" in out else f"✅ {out['new_signals']} new signals")

# --------------------------------------------------------------------- header
st.title("🛡️ Event-Driven Portfolio Stress Testing")
st.caption(
    "Module B subscribes to the **event class** and **impact score** from the AI/NLP "
    "Risk Engine. A high-impact adverse event triggers a scenario-specific shock to a "
    "synthetic wholesale book of loans, bonds, derivatives and equities.")

pf_df = portfolio.to_frame()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Positions", len(pf_df))
c2.metric("Net portfolio value", money(portfolio.value))
c3.metric("Gross exposure", money(portfolio.gross_exposure))
c4.metric("Capital buffer", money(portfolio.capital_buffer),
          f"{portfolio.capital_ratio:.0%} of gross")

tab_runs, tab_whatif, tab_book = st.tabs(
    ["📉 Triggered stress tests", "🎛️ What-if simulator", "📚 Portfolio"])


# --------------------------------------------------------------- shared views
def show_result(r: dict, key: str) -> None:
    color = STATUS_COLOR.get(r["status"], "#888")
    st.markdown(
        f"<div style='padding:10px 14px;border-left:6px solid {color};"
        f"background:rgba(128,128,128,0.10);border-radius:4px'>"
        f"<b>{r['headline']}</b></div>", unsafe_allow_html=True)
    st.write("")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Value before", money(r["value_before"]))
    m2.metric("Value after", money(r["value_after"]),
              f"{r['pnl_pct']:+.2f}%")
    m3.metric("Stress P&L", money(r["pnl"], signed=True))
    m4.metric("Capital buffer used", f"{r['buffer_used_pct']:.0f}%")
    m5.metric("Status", r["status"])

    left, right = st.columns(2)
    asset = pd.DataFrame(r["by_asset"]).T.reset_index(names="asset_type")
    fig = go.Figure()
    fig.add_bar(name="Before", x=asset["asset_type"], y=asset["before"] / 1e6,
                marker_color="#4c78a8")
    fig.add_bar(name="After stress", x=asset["asset_type"], y=asset["after"] / 1e6,
                marker_color="#e45756")
    fig.update_layout(barmode="group", title="Portfolio value before vs after ($M)",
                      yaxis_title="$M", height=380)
    left.plotly_chart(fig, width="stretch", key=f"{key}-asset")

    fac = r["by_factor"]
    names = [f for f, v in fac.items() if abs(v) > 1]
    wf = go.Figure(go.Waterfall(
        x=["Before"] + [n.upper() for n in names] + ["After"],
        measure=["absolute"] + ["relative"] * len(names) + ["total"],
        y=[r["value_before"] / 1e6] + [fac[n] / 1e6 for n in names] + [0],
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#4c78a8"}},
        connector={"line": {"color": "gray"}}))
    lo = min(r["value_before"], r["value_after"]) / 1e6
    hi = max(r["value_before"], r["value_after"]) / 1e6
    pad = max((hi - lo) * 0.6, 1.0)
    wf.update_layout(title="P&L bridge by risk factor ($M)", height=380,
                     yaxis_range=[lo - pad, hi + pad * 0.4], showlegend=False)
    right.plotly_chart(wf, width="stretch", key=f"{key}-wf")

    pos = pd.DataFrame(r["position_pnl"])
    l2, r2 = st.columns(2)
    heat = pos.pivot_table(index="sector", columns="asset_type", values="pnl",
                           aggfunc="sum", fill_value=0.0) / 1e6
    hf = px.imshow(heat, text_auto=".1f", aspect="auto",
                   color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                   title="P&L heatmap: sector × asset class ($M)")
    hf.update_layout(height=400)
    l2.plotly_chart(hf, width="stretch", key=f"{key}-heat")
    reg = pos.groupby("region")["pnl"].sum().sort_values() / 1e6
    rf = go.Figure(go.Bar(x=reg.values, y=reg.index, orientation="h",
                          marker_color=["#d62728" if v < 0 else "#2ca02c" for v in reg.values]))
    rf.update_layout(title="P&L by region ($M)", height=400)
    r2.plotly_chart(rf, width="stretch", key=f"{key}-reg")

    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Worst-hit positions**")
        top = pd.DataFrame(r["top_positions"])[
            ["trade_id", "counterparty", "asset_type", "instrument", "sector",
             "market_value", "pnl", "pnl_pct"]].copy()
        top["market_value"] = (top["market_value"] / 1e6).round(1)
        top["pnl"] = (top["pnl"] / 1e6).round(2)
        top["pnl_pct"] = top["pnl_pct"].round(2)
        top = top.rename(columns={"market_value": "MV $M", "pnl": "P&L $M", "pnl_pct": "P&L %"})
        st.dataframe(top, width="stretch", hide_index=True)
    with b:
        st.markdown("**Applied shocks** (scaled by impact ÷ 10)")
        sh = pd.DataFrame({"shock": list(r["shock"].keys()),
                           "value": [str(v) for v in r["shock"].values()]})
        st.dataframe(sh, width="stretch", hide_index=True)


# ----------------------------------------------------------------- tab: runs
with tab_runs:
    runs = results.list(limit=200, with_detail=False)
    if not runs:
        st.info("No stress tests yet. Click **Replay Kaggle datasets** in the sidebar — "
                "every adverse event with impact above the threshold will trigger one — "
                "or use the **What-if simulator** tab.")
    else:
        rdf = pd.DataFrame(runs)
        auto = rdf[rdf["trigger"] == "auto"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stress tests run", len(rdf))
        k2.metric("Auto-triggered by signals", len(auto))
        k3.metric("Worst loss", money(rdf["pnl"].min()))
        k4.metric("RED breaches", int((rdf["status"] == "RED").sum()))

        t = rdf.sort_values("id")
        tl = px.bar(t, x="id", y=t["pnl"] / 1e6, color="status",
                    color_discrete_map=STATUS_COLOR,
                    hover_data=["scenario_name", "event_class", "impact", "ticker"],
                    title="Stress P&L per triggered test ($M)",
                    labels={"id": "stress run #", "y": "P&L $M"})
        tl.update_layout(height=320)
        st.plotly_chart(tl, width="stretch")

        labels = {r["id"]: f"#{r['id']} · {r['scenario_name']} · impact {r['impact']} · "
                           f"{money(r['pnl'], signed=True)} · {r['status']}" for r in runs}
        pick = st.selectbox("Inspect a stress test", list(labels.keys()),
                            format_func=lambda i: labels[i])
        detail = results.get(pick)
        if detail:
            if detail.get("signal_id"):
                sig = store.get_signals(limit=2000)
                match = next((s for s in sig if s.signal_id == detail["signal_id"]), None)
                if match:
                    st.caption(f"**Triggering signal** ({match.source}): {match.text[:240]}")
            show_result(detail, key=f"run-{pick}")

        with st.expander("All runs"):
            cols = ["id", "ts", "trigger", "scenario_name", "event_class", "impact",
                    "ticker", "value_before", "value_after", "pnl", "pnl_pct",
                    "buffer_used_pct", "status"]
            st.dataframe(rdf[cols], width="stretch", hide_index=True)

# -------------------------------------------------------------- tab: what-if
with tab_whatif:
    st.markdown("Run any event through the same scenario engine the NLP signals use.")
    w1, w2, w3 = st.columns(3)
    ev = w1.selectbox("Event class", [e.value for e in EventClass if e is not EventClass.OTHER],
                      index=0)
    imp = w2.slider("Impact score", 1, 10, 8)
    tickers = ["(market-wide)"] + sorted(portfolio.tickers())
    tk = w3.selectbox("Named company (optional)", tickers)
    txt = ""
    if ev == EventClass.MACROECONOMIC.value:
        variant = st.radio("Macro variant", ["Recession / growth scare",
                                             "Inflation / rate-hike shock"], horizontal=True)
        txt = "inflation rate hike" if "rate" in variant else "recession slowdown"
    ticker = None if tk == "(market-wide)" else tk
    base = build_scenario(ev, imp, ticker, txt)

    with st.expander("Fine-tune the market shock (optional)"):
        s1, s2, s3 = st.columns(3)
        rates = s1.slider("Rates (bp)", -300, 300, int(base.market.rates_bp))
        spread = s2.slider("Credit spread (bp)", -50, 600, int(base.market.spread_bp))
        eq = s3.slider("Equity (%)", -50, 20, int(base.market.equity_pct * 100))
        s4, s5, s6 = st.columns(3)
        fxem = s4.slider("EM FX (%)", -30, 10, int(base.market.fx_em_pct * 100))
        vol = s5.slider("Implied vol (pts)", -10, 40, int(base.market.vol_pts))
        pdm = s6.slider("PD multiplier", 1.0, 8.0, float(round(base.market.pd_mult, 1)), 0.1)
        from module_b.scenarios import custom_scenario
        base = custom_scenario(base, rates_bp=float(rates), spread_bp=float(spread),
                               equity_pct=eq / 100.0, fx_em_pct=fxem / 100.0,
                               vol_pts=float(vol), pd_mult=float(pdm))

    st.caption(f"**{base.name}** — {base.description}")
    res = StressEngine(portfolio, cfg).run(base, trigger="manual").to_dict()
    show_result(res, key="whatif")

    if st.button("💾 Save this run to history", key="save-whatif"):
        res_id = results.add(res)
        st.success(f"Saved as stress run #{res_id}")

# ----------------------------------------------------------------- tab: book
with tab_book:
    p1, p2 = st.columns(2)
    by_type = pf_df.groupby("asset_type")["market_value"].sum().abs().reset_index()
    p1.plotly_chart(px.pie(by_type, names="asset_type", values="market_value", hole=0.45,
                           title="Exposure by asset class"), width="stretch")
    by_sec = (pf_df.assign(mv=pf_df["market_value"].abs() / 1e6)
              .groupby("sector", as_index=False)["mv"].sum().sort_values("mv"))
    p2.plotly_chart(px.bar(by_sec, x="mv", y="sector", orientation="h",
                           title="Exposure by sector ($M)", labels={"mv": "$M", "sector": ""}),
                    width="stretch")
    st.dataframe(pf_df, width="stretch", hide_index=True)
    st.caption("Loans: Kaggle 'Financial Transactions' (13M card transactions, aggregated into "
               "sector x region credit segments - data/processed/loan_book.csv). Bonds, derivatives and "
               "equities are a synthetic trading book - see module_b/portfolio.py.")