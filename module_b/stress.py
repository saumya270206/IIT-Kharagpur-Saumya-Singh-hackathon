"""Module B: strategic, event-driven portfolio stress testing.

``StressTester.apply(signal)`` is registered as a pipeline subscriber. When the
engine emits a high-impact adverse event it builds the matching scenario,
revalues the whole wholesale book, persists the before/after result and
returns it. ``run_scenario`` runs the same machinery on demand (what-if).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from risk_engine.models import EventClass, RiskSignal
from module_b.portfolio import Portfolio, load_portfolio
from module_b.scenarios import Scenario, build_scenario, scenario_from_signal
from module_b.store import StressStore
from module_b.valuation import FACTORS, position_shock, revalue

log = logging.getLogger("module_b.stress")


@dataclass(frozen=True)
class StressConfig:
    impact_threshold: int = 7       # trigger when impact is strictly greater than this
    max_sentiment: float = -0.10    # adverse events only (sentiment at or below this)
    # Event classes that are systemic shocks: stress-tested whenever impact is high,
    # whatever the sentiment (a war escalation is a portfolio risk even if one
    # company, e.g. an oil major, benefits from it).
    systemic_classes: tuple[str, ...] = (EventClass.GEOPOLITICAL.value,)
    # Suppress repeats of the same (class, company) story for this long (24h) -
    # unless the repeat is MORE severe than the run already done (escalation).
    cooldown_minutes: float = 1440.0
    min_confidence: float = 0.0
    # Market-wide macro commentary is common and noisy on social media: one tweet saying
    # "recession is coming" must not trigger a book-wide stress test. These classes need
    # `corroboration_min` qualifying (impact > threshold, adverse) signals inside the
    # cooldown window before the first stress test fires.
    corroboration_classes: tuple[str, ...] = (EventClass.MACROECONOMIC.value,)
    corroboration_min: int = 3
    skip_classes: tuple[str, ...] = (EventClass.OTHER.value,)
    amber_buffer_pct: float = 25.0  # RAG status on share of capital buffer consumed
    red_buffer_pct: float = 60.0

    def rule_text(self) -> str:
        sys_ = ", ".join(self.systemic_classes) or "none"
        return (f"auto stress test when impact > {self.impact_threshold} and "
                f"(sentiment <= {self.max_sentiment} or event class is systemic: {sys_}); "
                f"repeat stories suppressed for {self.cooldown_minutes / 60:g}h "
                f"unless impact escalates; "
                f"{', '.join(self.corroboration_classes) or 'none'} needs "
                f"{self.corroboration_min} corroborating signals")


@dataclass
class StressResult:
    run_id: Optional[int]
    ts: str
    trigger: str                    # "auto" | "manual"
    signal_id: Optional[str]
    headline: str
    scenario_key: str
    scenario_name: str
    event_class: str
    impact: int
    sentiment: Optional[float]
    ticker: Optional[str]
    value_before: float
    value_after: float
    pnl: float
    pnl_pct: float
    capital_buffer: float
    buffer_used_pct: float
    status: str                     # GREEN | AMBER | RED
    by_asset: dict = field(default_factory=dict)    # {type: {before, after, pnl}}
    by_factor: dict = field(default_factory=dict)   # {factor: pnl}
    by_sector: dict = field(default_factory=dict)
    by_region: dict = field(default_factory=dict)
    top_positions: list = field(default_factory=list)   # worst P&L first
    position_pnl: list = field(default_factory=list)    # every position
    shock: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _group(positions, pnl_by_trade, key) -> dict:
    out: dict[str, dict] = {}
    for p in positions:
        g = out.setdefault(getattr(p, key), {"before": 0.0, "after": 0.0, "pnl": 0.0})
        pnl = pnl_by_trade[p.trade_id]
        g["before"] += p.market_value
        g["after"] += p.market_value + pnl
        g["pnl"] += pnl
    return out


class StressEngine:
    """Pure function layer: (portfolio, scenario) -> StressResult. No I/O."""

    def __init__(self, portfolio: Portfolio, cfg: StressConfig | None = None) -> None:
        self.portfolio = portfolio
        self.cfg = cfg or StressConfig()

    def run(
        self,
        scenario: Scenario,
        trigger: str = "manual",
        signal: RiskSignal | None = None,
    ) -> StressResult:
        pf = self.portfolio
        pnl_by_trade: dict[str, float] = {}
        factors = {f: 0.0 for f in FACTORS}
        for p in pf.positions:
            comp = revalue(p, position_shock(p, scenario))
            pnl_by_trade[p.trade_id] = sum(comp.values())
            for f, v in comp.items():
                factors[f] += v

        before = pf.value
        pnl = sum(pnl_by_trade.values())
        after = before + pnl
        buffer = pf.capital_buffer
        used = (-pnl / buffer * 100.0) if buffer > 0 and pnl < 0 else 0.0
        status = ("RED" if used >= self.cfg.red_buffer_pct
                  else "AMBER" if used >= self.cfg.amber_buffer_pct else "GREEN")

        pos_rows = [
            {"trade_id": p.trade_id, "counterparty": p.counterparty,
             "asset_type": p.asset_type, "instrument": p.instrument,
             "sector": p.sector, "region": p.region,
             "market_value": p.market_value, "pnl": pnl_by_trade[p.trade_id],
             "pnl_pct": (pnl_by_trade[p.trade_id] / abs(p.market_value) * 100.0
                         if p.market_value else 0.0)}
            for p in pf.positions
        ]
        worst = sorted(pos_rows, key=lambda r: r["pnl"])[:10]
        sign = "-" if pnl < 0 else "+"
        headline = (f"{scenario.name} (impact {scenario.impact}/10): book "
                    f"{sign}${abs(pnl) / 1e6:,.1f}M ({pnl / abs(before) * 100:+.2f}%), "
                    f"{used:.0f}% of capital buffer")
        return StressResult(
            run_id=None,
            ts=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            signal_id=signal.signal_id if signal else None,
            headline=headline,
            scenario_key=scenario.key, scenario_name=scenario.name,
            event_class=scenario.event_class, impact=scenario.impact,
            sentiment=signal.sentiment if signal else None,
            ticker=scenario.ticker,
            value_before=before, value_after=after, pnl=pnl,
            pnl_pct=pnl / abs(before) * 100.0 if before else 0.0,
            capital_buffer=buffer, buffer_used_pct=used, status=status,
            by_asset=_group(pf.positions, pnl_by_trade, "asset_type"),
            by_factor=factors,
            by_sector=_group(pf.positions, pnl_by_trade, "sector"),
            by_region=_group(pf.positions, pnl_by_trade, "region"),
            top_positions=worst, position_pnl=pos_rows,
            shock=scenario.shock_summary(),
        )


class StressTester:
    """Pipeline subscriber: signal in -> (maybe) stress test out."""

    def __init__(
        self,
        store,
        portfolio: Portfolio | None = None,
        cfg: StressConfig | None = None,
        results: StressStore | None = None,
    ) -> None:
        self.cfg = cfg or StressConfig()
        self.portfolio = portfolio or load_portfolio()
        self.engine = StressEngine(self.portfolio, self.cfg)
        self.results = results or StressStore(store)
        self._last_trigger: dict[tuple[str, Optional[str]], tuple[datetime, int]] = {}
        self._recent: dict[str, list[datetime]] = {}   # corroboration history per class

    # ----------------------------------------------------------- triggering
    def should_trigger(self, signal: RiskSignal) -> tuple[bool, str]:
        c = self.cfg
        if signal.event_class.value in c.skip_classes:
            return False, "event class not stress-testable"
        if signal.impact <= c.impact_threshold:
            return False, f"impact {signal.impact} <= threshold {c.impact_threshold}"
        if (signal.sentiment > c.max_sentiment
                and signal.event_class.value not in c.systemic_classes):
            return False, "event is not adverse"
        if signal.confidence < c.min_confidence:
            return False, "low confidence"
        if signal.event_class.value in c.corroboration_classes:
            hist = self._recent.setdefault(signal.event_class.value, [])
            hist.append(signal.event_at)
            cutoff = signal.event_at - timedelta(minutes=c.cooldown_minutes)
            hist[:] = [t for t in hist if t >= cutoff]
            if len(hist) < c.corroboration_min:
                return False, f"awaiting corroboration ({len(hist)}/{c.corroboration_min})"
        key = (signal.event_class.value, signal.ticker)
        last = self._last_trigger.get(key)
        if last:
            last_at, last_impact = last
            if (abs(signal.event_at - last_at) < timedelta(minutes=c.cooldown_minutes)
                    and signal.impact <= last_impact):
                return False, "cooldown (same event already stress-tested)"
        return True, "triggered"

    def apply(self, signal: RiskSignal) -> Optional[dict]:
        ok, why = self.should_trigger(signal)
        if not ok:
            log.debug("no stress test for %s: %s", signal.signal_id, why)
            return None
        self._last_trigger[(signal.event_class.value, signal.ticker)] = (
            signal.event_at, signal.impact)
        result = self.engine.run(scenario_from_signal(signal), "auto", signal)
        result.run_id = self.results.add(result)
        log.info("stress test #%s: %s", result.run_id, result.headline)
        return result.to_dict()

    # ------------------------------------------------------------- what-if
    def run_scenario(
        self,
        event_class: EventClass | str,
        impact: int,
        ticker: Optional[str] = None,
        text: str = "",
        scenario: Scenario | None = None,
        persist: bool = True,
    ) -> StressResult:
        sc = scenario or build_scenario(event_class, impact, ticker, text)
        result = self.engine.run(sc, "manual")
        if persist:
            result.run_id = self.results.add(result)
        return result