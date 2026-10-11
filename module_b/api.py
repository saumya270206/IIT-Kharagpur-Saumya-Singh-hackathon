"""Module B HTTP routes, mounted onto the engine API by run_api.py."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from module_b.scenarios import TEMPLATES
from module_b.stress import StressTester
from risk_engine.models import EventClass


class WhatIfIn(BaseModel):
    event_class: EventClass = Field(description="e.g. Geopolitical, Macroeconomic")
    impact: int = Field(ge=1, le=10)
    ticker: Optional[str] = Field(default=None, description="name-specific event")
    text: str = Field(default="", description="optional text; steers macro variant")


def build_router(tester: StressTester) -> APIRouter:
    router = APIRouter(prefix="/stress", tags=["Module B - stress testing"])

    @router.get("/config")
    def config() -> dict:
        c = tester.cfg
        return {"impact_threshold": c.impact_threshold, "max_sentiment": c.max_sentiment,
                "cooldown_minutes": c.cooldown_minutes,
                "systemic_classes": list(c.systemic_classes),
                "rule": c.rule_text()}

    @router.get("/portfolio")
    def portfolio() -> dict:
        pf = tester.portfolio
        return {**pf.summary(), "trades": [p.to_dict() for p in pf.positions]}

    @router.get("/scenarios")
    def scenarios() -> list[dict]:
        return [{"key": t.key, "event_class": t.event_class.value,
                 "title": t.title, "description": t.description}
                for t in TEMPLATES.values()]

    @router.get("/runs")
    def runs(limit: int = Query(25, ge=1, le=500), detail: bool = False) -> list[dict]:
        return tester.results.list(limit=limit, with_detail=detail)

    @router.get("/runs/{run_id}")
    def run(run_id: int) -> dict:
        r = tester.results.get(run_id)
        if r is None:
            raise HTTPException(404, "unknown stress run")
        return r

    @router.post("/run", status_code=201)
    def what_if(body: WhatIfIn) -> dict:
        """Manual what-if: stress the book with any event class / impact."""
        res = tester.run_scenario(body.event_class, body.impact, body.ticker, body.text)
        return res.to_dict()

    return router