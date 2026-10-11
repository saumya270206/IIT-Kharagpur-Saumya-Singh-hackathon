"""AI/NLP Risk Engine page: try the engine live and inspect the signal stream."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import api_get, api_post
from module_b.scenarios import build_scenario
from risk_engine.models import RawItem
from risk_engine.nlp.local import LocalAnalyzer
from risk_engine.store import SignalStore

st.set_page_config(page_title="AI/NLP Risk Engine", layout="wide")
store = SignalStore()

st.sidebar.title("🧠 AI/NLP Risk Engine")
health = api_get("/health")
if health:
    st.sidebar.success(f"Online · {health['signals']} signals · backend: {health['nlp_backend']}")
else:
    st.sidebar.error("Engine API offline — start `python run_api.py`")

st.title("🧠 AI/NLP Risk Engine — Signal Lab")
st.caption("Unstructured text in → structured risk signal out "
           "(sentiment −1…1 · event class · impact 1…10 · ticker).")

EXAMPLES = {
    "Credit downgrade (JPM)": "JPMorgan misses estimates as Moody's downgrades its credit rating, sparking a selloff",
    "Geopolitical escalation": "Sanctions and military conflict escalate, oil surges and global markets plunge",
    "Hawkish Fed": "Fed signals aggressive rate hikes as inflation surges, yields spike",
    "Bullish earnings (NVDA)": "$NVDA smashes earnings, record revenue and raised guidance",
    "Tweet-style": "$TSLA keeps missing delivery estimates, this is a disaster for the stock",
}
choice = st.selectbox("Example", ["(write your own)"] + list(EXAMPLES))
text = st.text_area("Text to analyze (news headline, tweet, …)",
                    value=EXAMPLES.get(choice, ""), height=90)

if text.strip():
    sig = LocalAnalyzer().analyze(RawItem(id="preview", source="manual", text=text))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", f"{sig.sentiment:+.2f}")
    c2.metric("Event class", sig.event_class.value)
    c3.metric("Impact", f"{sig.impact}/10")
    c4.metric("Entity", sig.ticker or "—", sig.company or None)
    st.json(sig.to_json(), expanded=False)

    sc = build_scenario(sig.event_class, sig.impact, sig.ticker, sig.text)
    would = sig.impact > 7 and sig.sentiment <= -0.10
    if sig.ticker:
        a_txt = (f"Module A would **{'raise' if sig.sentiment > 0 else 'cut'}** "
                 f"{sig.ticker}'s index weight" if abs(sig.sentiment) > 0.01
                 else f"Module A would leave {sig.ticker} unchanged (neutral)")
    else:
        a_txt = "Module A would ignore it (no index constituent named)"
    b_txt = (f"Module B **would trigger** the *{sc.name}* stress test." if would
             else "Module B would **not** stress-test (needs impact > 7 and adverse sentiment).")
    st.markdown(f"**Downstream:** {a_txt} · {b_txt}")

    if st.button("🚀 Publish to the live system (updates weights + stress tests)"):
        out = api_post("/signals", {"text": text, "source": "manual"})
        st.write(out if "error" in out else "✅ Published — see the other pages.")

st.divider()
sigs = store.get_signals(limit=1000)
if not sigs:
    st.info("No signals stored yet — ingest the sample corpus from the Module A page.")
    st.stop()

df = pd.DataFrame([s.to_json() for s in sigs])
df["event_class"] = df["event_class"].astype(str)
a, b, c = st.columns(3)
a.plotly_chart(px.histogram(df, x="event_class", color="source",
                            title="Signals by event class & source").update_layout(height=340),
               width="stretch")
b.plotly_chart(px.histogram(df, x="sentiment", nbins=20, color="source",
                            title="Sentiment distribution").update_layout(height=340),
               width="stretch")
c.plotly_chart(px.box(df, x="event_class", y="impact", points="all",
                      title="Impact by event class").update_layout(height=340),
               width="stretch")
st.subheader("Signal stream")
st.dataframe(df[["created_at", "source", "ticker", "event_class", "sentiment", "impact",
                 "confidence", "backend", "text"]].head(100),
             width="stretch", hide_index=True)
