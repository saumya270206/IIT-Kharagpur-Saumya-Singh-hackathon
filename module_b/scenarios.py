"""Scenario library: maps an engine signal (event class + impact) to market shocks.

Every template is calibrated at **impact = 10** (loosely anchored on historical
episodes: Mar-2020, 2008, Mar-2022) and scaled linearly by ``impact / 10``, so an
impact-8 geopolitical event applies 80% of the reference shock.

Two layers per template:
  * ``market``  - systemic shock (rates, spreads, equity, FX, vol, PD) applied to everyone
  * ``idio``    - name-specific shock for the company in the signal (if the
                  signal names one), with a fraction spilling over to its sector

``named_market_leak`` is how much of the systemic layer still applies when the
signal names a single company (a JPM downgrade moves the market a little; a war
moves it fully).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

from risk_engine.models import EventClass, RiskSignal
from module_b.portfolio import SECTOR_BY_TICKER


@dataclass(frozen=True)
class MarketShock:
    rates_bp: float = 0.0          # parallel yield shift, bp (negative = rally)
    spread_bp: float = 0.0         # BBB-equivalent credit spread widening, bp
    equity_pct: float = 0.0        # broad equity return (-0.18 = -18%)
    fx_major_pct: float = 0.0      # EUR/GBP/JPY... vs USD
    fx_em_pct: float = 0.0         # EM currencies vs USD
    vol_pts: float = 0.0           # implied-vol change, vol points
    pd_mult: float = 1.0           # multiplier on probability of default
    sector_equity: Mapping[str, float] = field(default_factory=dict)
    sector_spread_bp: Mapping[str, float] = field(default_factory=dict)

    def scaled(self, k: float) -> "MarketShock":
        return MarketShock(
            rates_bp=self.rates_bp * k, spread_bp=self.spread_bp * k,
            equity_pct=self.equity_pct * k, fx_major_pct=self.fx_major_pct * k,
            fx_em_pct=self.fx_em_pct * k, vol_pts=self.vol_pts * k,
            pd_mult=1.0 + (self.pd_mult - 1.0) * k,
            sector_equity={s: v * k for s, v in self.sector_equity.items()},
            sector_spread_bp={s: v * k for s, v in self.sector_spread_bp.items()},
        )


@dataclass(frozen=True)
class IdioShock:
    equity_pct: float = 0.0
    spread_bp: float = 0.0
    pd_mult: float = 1.0
    sector_contagion: float = 0.0  # share of the idio shock hitting same-sector names

    def scaled(self, k: float) -> "IdioShock":
        return IdioShock(
            equity_pct=self.equity_pct * k, spread_bp=self.spread_bp * k,
            pd_mult=1.0 + (self.pd_mult - 1.0) * k,
            sector_contagion=self.sector_contagion,
        )


@dataclass(frozen=True)
class Template:
    key: str
    event_class: EventClass
    title: str
    description: str
    market: MarketShock = field(default_factory=MarketShock)
    idio: IdioShock = field(default_factory=IdioShock)
    named_market_leak: float = 1.0


@dataclass(frozen=True)
class Scenario:
    """A template instantiated for one concrete event (scaled, resolved)."""

    key: str
    name: str
    description: str
    event_class: str
    impact: int
    severity: float
    market: MarketShock
    idio: IdioShock
    ticker: Optional[str] = None
    sector: Optional[str] = None

    def shock_summary(self) -> dict:
        m = self.market
        out = {
            "rates_bp": round(m.rates_bp, 1), "spread_bp": round(m.spread_bp, 1),
            "equity_pct": round(m.equity_pct * 100, 2),
            "fx_major_pct": round(m.fx_major_pct * 100, 2),
            "fx_em_pct": round(m.fx_em_pct * 100, 2),
            "vol_pts": round(m.vol_pts, 1), "pd_mult": round(m.pd_mult, 2),
        }
        if self.ticker:
            out["idio_ticker"] = self.ticker
            out["idio_equity_pct"] = round(self.idio.equity_pct * 100, 2)
            out["idio_spread_bp"] = round(self.idio.spread_bp, 1)
        return out


# --------------------------------------------------------------------------
# Template library (reference severity = impact 10)
# --------------------------------------------------------------------------
TEMPLATES: dict[str, Template] = {t.key: t for t in [
    Template(
        "geopolitical", EventClass.GEOPOLITICAL, "Geopolitical shock",
        "Flight to quality: equities sell off, oil spikes, EM currencies weaken, "
        "credit spreads widen while government yields fall.",
        market=MarketShock(
            rates_bp=-50, spread_bp=150, equity_pct=-0.18, fx_major_pct=-0.03,
            fx_em_pct=-0.10, vol_pts=14, pd_mult=2.0,
            sector_equity={"Energy": 0.26, "Technology": -0.04},
            sector_spread_bp={"Energy": -90, "Industrials": 40,
                              "Consumer Discretionary": 40}),
        named_market_leak=1.0),
    Template(
        "macro_rates", EventClass.MACROECONOMIC, "Macro: inflation / rate-hike shock",
        "Hawkish central bank repricing: yields jump, long-duration assets and "
        "growth equities fall, credit spreads and EM FX come under pressure.",
        market=MarketShock(
            rates_bp=150, spread_bp=100, equity_pct=-0.15, fx_major_pct=-0.02,
            fx_em_pct=-0.07, vol_pts=10, pd_mult=1.8,
            sector_equity={"Technology": -0.06, "Utilities": -0.04, "Financials": 0.03}),
        named_market_leak=1.0),
    Template(
        "macro_recession", EventClass.MACROECONOMIC, "Macro: recession / growth scare",
        "Growth shock: yields rally but credit spreads gap wider, cyclical equities "
        "fall hard and default expectations rise.",
        market=MarketShock(
            rates_bp=-100, spread_bp=250, equity_pct=-0.25, fx_major_pct=-0.02,
            fx_em_pct=-0.08, vol_pts=18, pd_mult=3.0,
            sector_equity={"Consumer Discretionary": -0.08, "Industrials": -0.05,
                           "Financials": -0.05, "Consumer Staples": 0.06,
                           "Healthcare": 0.04},
            sector_spread_bp={"Financials": 60, "Consumer Discretionary": 80,
                              "Energy": 40}),
        named_market_leak=1.0),
    Template(
        "credit_event", EventClass.CREDIT_EVENT, "Credit event / downgrade",
        "Default or downgrade of a counterparty with contagion through financial "
        "sector spreads and a modest systemic risk-off.",
        market=MarketShock(
            rates_bp=-30, spread_bp=250, equity_pct=-0.12, vol_pts=12, pd_mult=3.0,
            sector_equity={"Financials": -0.10},
            sector_spread_bp={"Financials": 120}),
        idio=IdioShock(equity_pct=-0.30, spread_bp=400, pd_mult=6.0,
                       sector_contagion=0.35),
        named_market_leak=0.25),
    Template(
        "regulatory", EventClass.REGULATORY, "Regulatory action",
        "Fine, probe or adverse rule change: name-specific equity and spread "
        "damage with some sector read-across.",
        market=MarketShock(equity_pct=-0.03, spread_bp=20, vol_pts=3, pd_mult=1.1),
        idio=IdioShock(equity_pct=-0.15, spread_bp=80, pd_mult=1.8,
                       sector_contagion=0.40),
        named_market_leak=0.10),
    Template(
        "operational", EventClass.OPERATIONAL, "Operational / cyber incident",
        "Outage, breach or recall: sharp but mostly company-specific repricing.",
        market=MarketShock(equity_pct=-0.02, spread_bp=15, vol_pts=2, pd_mult=1.05),
        idio=IdioShock(equity_pct=-0.12, spread_bp=70, pd_mult=1.6,
                       sector_contagion=0.20),
        named_market_leak=0.10),
    Template(
        "earnings", EventClass.EARNINGS, "Earnings shock",
        "Large miss / guidance cut: idiosyncratic equity gap and spread widening.",
        market=MarketShock(equity_pct=-0.02, spread_bp=10, vol_pts=2),
        idio=IdioShock(equity_pct=-0.14, spread_bp=60, pd_mult=1.5,
                       sector_contagion=0.20),
        named_market_leak=0.10),
    Template(
        "mna", EventClass.MERGER_ACQUISITION, "M&A disruption",
        "Deal break or leveraged takeover: acquirer equity and credit repricing.",
        market=MarketShock(equity_pct=-0.01, spread_bp=5),
        idio=IdioShock(equity_pct=-0.08, spread_bp=40, pd_mult=1.3,
                       sector_contagion=0.10),
        named_market_leak=0.0),
    Template(
        "product", EventClass.PRODUCT_LAUNCH, "Product launch failure",
        "Failed launch / delay: contained, company-specific hit.",
        idio=IdioShock(equity_pct=-0.06, spread_bp=25, sector_contagion=0.10),
        named_market_leak=0.0),
    Template(
        "generic", EventClass.OTHER, "Generic adverse event",
        "Unclassified adverse event: mild broad risk-off.",
        market=MarketShock(equity_pct=-0.05, spread_bp=40, vol_pts=4, pd_mult=1.2),
        named_market_leak=0.5),
]}

_BY_CLASS = {t.event_class: t for t in TEMPLATES.values()
             if t.key not in ("macro_recession",)}
_RATE_HIKE_RE = re.compile(
    r"\b(hike|hikes|hiking|inflation|hawkish|tighten\w*|yields? (?:surge|spike|jump|soar)\w*"
    r"|rate (?:rise|increase)\w*|stagflation)\b", re.IGNORECASE)


def pick_template(event_class: EventClass | str, text: str = "") -> Template:
    ec = EventClass(event_class)
    if ec is EventClass.MACROECONOMIC:
        return TEMPLATES["macro_rates" if _RATE_HIKE_RE.search(text or "")
                         else "macro_recession"]
    return _BY_CLASS.get(ec, TEMPLATES["generic"])


def build_scenario(
    event_class: EventClass | str,
    impact: int,
    ticker: Optional[str] = None,
    text: str = "",
    template_key: Optional[str] = None,
) -> Scenario:
    """Instantiate the right template for an event, scaled by impact (1-10)."""
    impact = int(max(1, min(10, impact)))
    tpl = TEMPLATES[template_key] if template_key else pick_template(event_class, text)
    k = impact / 10.0
    ticker = ticker.upper() if ticker else None
    leak = tpl.named_market_leak if ticker else 1.0
    market = tpl.market.scaled(k * leak)
    idio = tpl.idio.scaled(k) if ticker else IdioShock()
    return Scenario(
        key=tpl.key,
        name=tpl.title + (f" - {ticker}" if ticker else ""),
        description=tpl.description,
        event_class=EventClass(event_class).value,
        impact=impact, severity=k, market=market, idio=idio, ticker=ticker,
        sector=SECTOR_BY_TICKER.get(ticker) if ticker else None,
    )


def scenario_from_signal(signal: RiskSignal) -> Scenario:
    # A favourable story for the named company (e.g. an oil major on a Middle East
    # escalation) must not get that company's *adverse* idiosyncratic shock; the
    # systemic market shock still applies.
    ticker = signal.ticker if signal.sentiment <= 0 else None
    return build_scenario(signal.event_class, signal.impact, ticker, signal.text)


def custom_scenario(
    base: Scenario,
    **overrides: float,
) -> Scenario:
    """Return ``base`` with selected market shock fields replaced (what-if sliders)."""
    market = replace(base.market, **{k: v for k, v in overrides.items()
                                     if hasattr(base.market, k)})
    return replace(base, market=market, name=base.name + " (custom)")