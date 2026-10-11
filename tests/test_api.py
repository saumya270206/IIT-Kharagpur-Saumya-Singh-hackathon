"""HTTP API: engine routes + Module A weights + Module B stress routes."""

import pytest
from fastapi.testclient import TestClient

from module_a import IndexRebalancer
from module_b import StressTester
from module_b.api import build_router
from risk_engine import config
from risk_engine.api import create_app
from risk_engine.nlp.analyzer import RiskAnalyzer
from risk_engine.pipeline import Pipeline
from risk_engine.store import SignalStore


@pytest.fixture
def client(tmp_path):
    store = SignalStore(tmp_path / "api.db")
    pipe = Pipeline(store, RiskAnalyzer(backend="off"))
    reb, stress = IndexRebalancer(store), StressTester(store)
    pipe.subscribe(reb.apply)
    pipe.subscribe(stress.apply)
    app = create_app(pipe, store)
    app.include_router(build_router(stress))
    return TestClient(app)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["subscribers"] == 2 and r["nlp_backend"] == "local"


def test_empty_state_404s(client):
    assert client.get("/signals/latest").status_code == 404
    assert client.get("/weights/latest").status_code == 404


def test_post_signal_updates_weights_and_returns_structured_output(client):
    r = client.post("/signals", json={"text": "Tesla smashes earnings with record revenue surge"})
    assert r.status_code == 201
    body = r.json()
    sig = body["signal"]
    assert sig["ticker"] == "TSLA" and sig["sentiment"] > 0
    assert -1 <= sig["sentiment"] <= 1 and 1 <= sig["impact"] <= 10 and sig["event_class"]
    assert body["weights"]["TSLA"] > 1 / 15
    assert client.get("/weights/latest").json()["sum"] == pytest.approx(1.0)
    assert len(client.get("/weights/history").json()) == 15


def test_pipeline_run_and_signal_queries(client, monkeypatch):
    monkeypatch.setattr(config, "DATASET_TWEET_LIMIT", 300)
    monkeypatch.setattr(config, "DATASET_NEWS_LIMIT", 100)
    out = client.post("/pipeline/run", json={"sources": ["datasets"]}).json()
    assert out["new_signals"] > 20
    assert client.post("/pipeline/run", json={"sources": ["nope"]}).status_code == 422
    assert len(client.get("/signals", params={"limit": 5}).json()) == 5
    only = client.get("/signals", params={"ticker": "TSLA"}).json()
    assert only and all(s["ticker"] == "TSLA" for s in only)
    runs = client.get("/stress/runs").json()
    assert runs and all(r["impact"] > 7 and r["trigger"] == "auto" for r in runs)


def test_module_b_routes(client):
    assert client.get("/stress/config").json()["impact_threshold"] == 7
    pf = client.get("/stress/portfolio").json()
    assert pf["positions"] >= 50 and pf["trades"]
    assert len(client.get("/stress/scenarios").json()) >= 9
    r = client.post("/stress/run", json={"event_class": "Geopolitical", "impact": 9})
    assert r.status_code == 201 and r.json()["pnl"] < 0
    rid = r.json()["run_id"]
    assert client.get(f"/stress/runs/{rid}").json()["by_factor"]
    assert client.get("/stress/runs/9999").status_code == 404
    assert client.post("/stress/run", json={"event_class": "Nope", "impact": 5}).status_code == 422
