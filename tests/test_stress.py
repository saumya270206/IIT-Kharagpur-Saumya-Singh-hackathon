"""Module B: scenarios, valuation, trigger rule, persistence."""

import pytest

from module_b import (
    StressConfig, StressEngine, StressStore, StressTester, build_scenario, load_portfolio,
)
from module_b.portfolio import Position, from_transactions_df, generate_transactions
from module_b.valuation import position_shock, revalue
from risk_engine.models import EventClass, RiskSignal
from risk_engine.store import SignalStore
from datetime import datetime, timedelta, timezone

_n = 0


def sig(event, impact, sentiment=-0.9, ticker=None, at=None):
    global _n
    _n += 1
    return RiskSignal(signal_id=f"s{_n}", item_id=f"i{_n}", source="news", text="x",
                      ticker=ticker, sentiment=sentiment, event_class=event, impact=impact,
                      event_at=at or datetime.now(timezone.utc))


@pytest.fixture(scope="module")
def pf():
    return load_portfolio()


@pytest.fixture
def tester(tmp_path, pf):
    return StressTester(SignalStore(tmp_path / "s.db"), pf)


def test_portfolio_has_all_asset_classes(pf):
    df = pf.to_frame()
    assert set(df.asset_type) == {"Loan", "Bond", "Derivative", "Equity"}
    assert {"IRS", "FX Forward"} <= set(df.instrument)
    assert len(pf.positions) >= 50 and pf.capital_buffer > 0


def test_generator_is_deterministic():
    assert generate_transactions(7).equals(generate_transactions(7))


def test_loader_accepts_foreign_columns():
    import pandas as pd
    df = pd.DataFrame({"transaction_id": ["a", "b"], "amount": [1e6, -2e6],
                       "merchant": ["Acme", "Beta"], "type": ["bond", "garbage"]})
    p = from_transactions_df(df)
    assert [x.asset_type for x in p.positions] == ["Bond", "Loan"]
    assert all(x.notional > 0 for x in p.positions)


def test_shock_scales_with_impact():
    lo = build_scenario(EventClass.GEOPOLITICAL, 4)
    hi = build_scenario(EventClass.GEOPOLITICAL, 8)
    assert hi.market.equity_pct == pytest.approx(2 * lo.market.equity_pct)
    assert hi.market.spread_bp == pytest.approx(2 * lo.market.spread_bp)


def test_macro_variant_follows_text():
    assert build_scenario("Macroeconomic", 9, text="Fed signals aggressive rate hikes").key == "macro_rates"
    assert build_scenario("Macroeconomic", 9, text="recession fears grow").key == "macro_recession"


def test_bigger_impact_means_bigger_loss(pf):
    eng = StressEngine(pf)
    losses = [eng.run(build_scenario(EventClass.GEOPOLITICAL, i)).pnl for i in (3, 6, 9)]
    assert losses[0] > losses[1] > losses[2]


@pytest.mark.parametrize("ev", [e for e in EventClass if e is not EventClass.OTHER])
def test_every_event_class_runs_and_reconciles(pf, ev):
    r = StressEngine(pf).run(build_scenario(ev, 9, "JPM"))
    assert r.value_after == pytest.approx(r.value_before + r.pnl)
    assert sum(r.by_factor.values()) == pytest.approx(r.pnl)
    assert sum(v["pnl"] for v in r.by_asset.values()) == pytest.approx(r.pnl)
    assert r.status in {"GREEN", "AMBER", "RED"}


def test_named_credit_event_hits_that_counterparty_hardest(pf):
    r = StressEngine(pf).run(build_scenario(EventClass.CREDIT_EVENT, 10, "BAC"))
    eq = {p["counterparty"]: p["pnl_pct"] for p in r.position_pnl
          if p["asset_type"] == "Equity"}
    # named name (-30% idio) > same-sector contagion (JPM) > unrelated sector (Microsoft)
    assert eq["Bank of America"] < eq["JPMorgan Chase"] < eq["Microsoft"] < 0


def test_rates_up_hurts_bonds_and_helps_pay_fixed_swaps():
    bond = Position("b", "Bond", "Corporate Bond", "X", "Technology", "NA", "USD", "A",
                    10e6, 10e6, 8.0, 4.0)
    pay = Position("s", "Derivative", "IRS", "X", "Technology", "NA", "USD", "A",
                   50e6, 0.0, 5.0, 3.0, direction=-1)
    sc = build_scenario("Macroeconomic", 10, text="inflation rate hike")
    b = sum(revalue(bond, position_shock(bond, sc)).values())
    s = revalue(pay, position_shock(pay, sc))
    assert b < 0 and s["rates"] > 0


def test_cash_instruments_cannot_lose_more_than_their_value():
    eq = Position("e", "Equity", "Equity", "X", "Technology", "NA", "USD", "A",
                  1e6, 1e6, 0.0)
    sc = build_scenario("Credit Event", 10, "AAPL")
    assert sum(revalue(eq, position_shock(eq, sc)).values()) >= -1e6 - 1e-6


def test_long_put_gains_when_equities_crash():
    put = Position("p", "Derivative", "Equity Put", "X", "Technology", "NA", "USD", "A",
                   20e6, 1e6, 1.0)
    sc = build_scenario("Geopolitical", 10)
    assert sum(revalue(put, position_shock(put, sc)).values()) > 0


# --------------------------------------------------------------- trigger rule
def test_triggers_on_high_impact_adverse_event(tester):
    out = tester.apply(sig(EventClass.GEOPOLITICAL, 9))
    assert out is not None and out["pnl"] < 0 and out["run_id"] == 1
    assert tester.results.count() == 1


@pytest.mark.parametrize("impact,sentiment,event", [
    (7, -0.9, EventClass.GEOPOLITICAL),      # impact must be strictly > 7
    (10, 0.8, EventClass.MACROECONOMIC),     # positive non-systemic news is not stressed
    (10, -0.9, EventClass.OTHER),            # unclassified
])
def test_does_not_trigger(tester, impact, sentiment, event):
    assert tester.apply(sig(event, impact, sentiment)) is None
    assert tester.results.count() == 0


def test_cooldown_suppresses_repeats_but_not_new_events(tester):
    t0 = datetime.now(timezone.utc)
    assert tester.apply(sig(EventClass.CREDIT_EVENT, 9, ticker="JPM", at=t0)) is not None
    assert tester.apply(sig(EventClass.CREDIT_EVENT, 9, ticker="JPM",
                            at=t0 + timedelta(minutes=5))) is None
    assert tester.apply(sig(EventClass.CREDIT_EVENT, 9, ticker="BAC",
                            at=t0 + timedelta(minutes=6))) is not None
    # still inside the 24h cooldown: same story, same severity -> suppressed
    assert tester.apply(sig(EventClass.CREDIT_EVENT, 9, ticker="JPM",
                            at=t0 + timedelta(hours=12))) is None
    # an escalation (higher impact) is stress-tested even inside the cooldown
    assert tester.apply(sig(EventClass.CREDIT_EVENT, 10, ticker="JPM",
                            at=t0 + timedelta(hours=13))) is not None
    # after the cooldown expires the story can be stress-tested again
    assert tester.apply(sig(EventClass.CREDIT_EVENT, 9, ticker="JPM",
                            at=t0 + timedelta(hours=40))) is not None


def test_positive_geopolitical_event_still_stressed_without_company_shock(tester):
    """High-impact geopolitical events are systemic: stressed even if sentiment > 0."""
    out = tester.apply(sig(EventClass.GEOPOLITICAL, 10, 0.95, ticker="XOM"))
    assert out is not None and out["pnl"] < 0
    assert out["ticker"] is None          # no adverse idiosyncratic shock on the beneficiary


def test_low_impact_positive_geopolitical_not_stressed(tester):
    assert tester.apply(sig(EventClass.GEOPOLITICAL, 7, 0.95)) is None


def test_custom_threshold(tmp_path, pf):
    t = StressTester(SignalStore(tmp_path / "c.db"), pf, StressConfig(impact_threshold=4))
    assert t.apply(sig(EventClass.REGULATORY, 5)) is not None


def test_results_persist_with_full_detail(tester):
    tester.apply(sig(EventClass.GEOPOLITICAL, 9))
    row = StressStore(tester.results._s).latest()
    assert row["by_factor"] and row["position_pnl"] and row["shock"]["pd_mult"] > 1
    assert len(tester.results.list(limit=5)) == 1


def test_what_if_manual_run(tester):
    res = tester.run_scenario("Geopolitical", 8)
    assert res.trigger == "manual" and res.run_id and res.pnl < 0

def test_macro_needs_corroboration_before_stress_test(tester):
    t0 = datetime(2022, 3, 1, 12, tzinfo=timezone.utc)
    assert tester.apply(sig(EventClass.MACROECONOMIC, 9, at=t0)) is None
    assert tester.apply(sig(EventClass.MACROECONOMIC, 9, at=t0 + timedelta(hours=1))) is None
    assert tester.apply(sig(EventClass.MACROECONOMIC, 9, at=t0 + timedelta(hours=2))) is not None
    # signals more than 24h apart never corroborate each other
    assert tester.apply(sig(EventClass.MACROECONOMIC, 9, at=t0 + timedelta(days=5))) is None
