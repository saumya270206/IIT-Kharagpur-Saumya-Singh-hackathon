"""HTTP API exposing the engine's structured risk signals.

The engine is application-agnostic: downstream modules subscribe via the
pipeline's handler callbacks. run_api.py wires Module A's rebalancer in, so
POST /signals both creates a signal and triggers the downstream reaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from risk_engine.models import RawItem, RiskSignal, utcnow
from risk_engine.pipeline import Pipeline, SOURCES
from risk_engine.store import SignalStore


class SignalIn(BaseModel):
    text: str = Field(min_length=3, description="Unstructured text to analyze")
    source: str = "manual"
    ticker: Optional[str] = None
    company: Optional[str] = None
    url: Optional[str] = None


class CycleIn(BaseModel):
    sources: list[str] = Field(default=["datasets"], description="datasets and/or live feeds (gdelt, rss, ...)")


def create_app(
    pipeline: Pipeline | None = None,
    store: SignalStore | None = None,
) -> FastAPI:
    store = store or SignalStore()
    pipeline = pipeline or Pipeline(store)

    app = FastAPI(
        title="AI/NLP Risk Engine",
        description="Structured risk signals (sentiment / event class / impact) "
                    "from unstructured news & social media.",
        version="1.0.0",
    )
    app.state.store = store
    app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "signals": store.count_signals(),
            "nlp_backend": pipeline.analyzer.backend,
            "subscribers": len(pipeline.handlers),
        }

    # ---------------------------------------------------------------- signals
    @app.get("/signals")
    def list_signals(
        limit: int = Query(50, ge=1, le=1000),
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[RiskSignal]:
        return store.get_signals(limit=limit, ticker=ticker, source=source, since=since)

    @app.get("/signals/latest")
    def latest_signal() -> RiskSignal:
        sigs = store.get_signals(limit=1)
        if not sigs:
            raise HTTPException(404, "no signals yet - run an ingest cycle first")
        return sigs[0]

    @app.post("/signals", status_code=201)
    def post_signal(body: SignalIn) -> dict:
        """Analyze ad-hoc text, persist the signal, notify subscribers."""
        item = RawItem(
            id=f"manual-{utcnow().timestamp()}",
            source=body.source,
            text=body.text,
            url=body.url,
            published_at=utcnow(),
            ticker=body.ticker,
            company=body.company,
        )
        signal = pipeline.process(item)
        if signal is None:
            raise HTTPException(409, "duplicate signal")
        return {"signal": signal, "weights": store.latest_weights()}

    # ---------------------------------------------------------------- weights
    @app.get("/weights/latest")
    def latest_weights() -> dict:
        weights = store.latest_weights()
        if not weights:
            raise HTTPException(404, "no rebalance has happened yet")
        return {"weights": weights, "sum": sum(weights.values()), "n": len(weights)}

    @app.get("/weights/history")
    def weights_history(since: Optional[datetime] = None) -> list[dict]:
        return store.get_weight_history(since=since)

    # ---------------------------------------------------------------- pipeline
    @app.post("/pipeline/run")
    def run_cycle(body: CycleIn) -> dict:
        unknown = [s for s in body.sources if s not in SOURCES]
        if unknown:
            raise HTTPException(422, f"unknown sources: {unknown}")
        return pipeline.run_cycle(body.sources)

    return app
