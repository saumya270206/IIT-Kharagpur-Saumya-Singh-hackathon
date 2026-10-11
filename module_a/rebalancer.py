"""Sentiment-driven tactical rebalancer (Module A).

Subscribes to RiskSignals from the engine:

  * positive sentiment -> increase the stock's weight
  * negative sentiment -> decrease the stock's weight
  * impact scales the size of the move (1-10)

Guard rails: a per-event turnover cap, min/max position bounds, and a mild
mean-reversion drift back toward equal weight so the index cannot lock into
extreme allocations over long runs. Every rebalance writes a full weight
snapshot (all tickers) to the shared store, which is what the dashboard
plots over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from risk_engine.models import RiskSignal
from risk_engine.store import SignalStore
from module_a.index import INDEX_TICKERS


@dataclass(frozen=True)
class RebalanceConfig:
    sensitivity: float = 0.60   # weight shift (x avg weight) at |sentiment|=1, impact=10
    reversion: float = 0.05     # per-event drift back toward equal weight
    min_weight: float = 0.005   # floor per constituent
    max_weight: float = 0.35    # cap per constituent
    turnover_cap: float = 0.08  # max one-way turnover per event (share of index)


class IndexRebalancer:
    """Holds current index weights and updates them from RiskSignals."""

    def __init__(
        self,
        store: SignalStore,
        config: RebalanceConfig | None = None,
        tickers: Sequence[str] = INDEX_TICKERS,
    ) -> None:
        self.store = store
        self.cfg = config or RebalanceConfig()
        self.tickers = list(tickers)
        self.weights: dict[str, float] = self._load_state()
        self._last_ts: datetime | None = self._load_last_ts()

    # ------------------------------------------------------------------ state
    @property
    def equal_weight(self) -> float:
        return 1.0 / len(self.tickers)

    def _load_state(self) -> dict[str, float]:
        """Resume from the last stored snapshot, else start equal-weighted."""
        stored = self.store.latest_weights()
        restored = {t: float(stored.get(t, 0.0)) for t in self.tickers if t in stored}
        if len(restored) == len(self.tickers) and abs(sum(restored.values()) - 1.0) < 0.05:
            return self._normalize(restored)
        return {t: self.equal_weight for t in self.tickers}

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _load_last_ts(self) -> datetime | None:
        times = self.store.snapshot_times()
        if not times:
            return None
        return datetime.fromisoformat(times[-1])

    def current_weights(self) -> dict[str, float]:
        return dict(self.weights)

    def reset(self) -> dict[str, float]:
        self.weights = {t: self.equal_weight for t in self.tickers}
        return self.current_weights()

    # ------------------------------------------------------------------ apply
    def apply(self, signal: RiskSignal) -> Optional[dict]:
        """Process one signal. Returns rebalance info, or None if out of scope
        (unresolved entity or a ticker outside the mock index)."""
        ticker = signal.ticker
        if ticker not in self.weights:
            return None

        cfg = self.cfg
        avg = self.equal_weight
        old = dict(self.weights)

        # target move: sign of sentiment, scaled by impact and sensitivity
        score = signal.sentiment * (signal.impact / 10.0)
        deltas = {t: 0.0 for t in self.tickers}
        deltas[ticker] = cfg.sensitivity * score * avg

        # mean reversion: everyone drifts a little toward equal weight
        for t in self.tickers:
            deltas[t] += -(old[t] - avg) * cfg.reversion

        # zero-sum: the trade must be self-funding, otherwise renormalizing
        # to 100% silently moves every other name and doubles real turnover
        mean_delta = sum(deltas.values()) / len(deltas)
        deltas = {t: d - mean_delta for t, d in deltas.items()}

        # per-event turnover cap (value bought == value sold, share of index)
        buys = sum(d for d in deltas.values() if d > 0)
        if buys > cfg.turnover_cap and buys > 0:
            scale = cfg.turnover_cap / buys
            deltas = {t: d * scale for t, d in deltas.items()}

        # apply, clamp to bounds, renormalize to exactly 100%
        new = {t: old[t] + deltas[t] for t in self.tickers}
        new = {t: min(cfg.max_weight, max(cfg.min_weight, v)) for t, v in new.items()}
        new = self._normalize(new)
        for _ in range(2):  # renorm can nudge a bound; two passes converge
            clamped = {t: min(cfg.max_weight, max(cfg.min_weight, v)) for t, v in new.items()}
            if clamped == new:
                break
            new = self._normalize(clamped)
        self.weights = new

        # full snapshot (all tickers) so the dashboard can plot weights over time.
        # snapshot time is clamped forward so wall-clock order == insertion order
        # (a stale headline's publication date must not move the timeline back).
        ts = signal.event_at
        if self._last_ts is not None and ts <= self._last_ts:
            ts = self._last_ts + timedelta(seconds=1)
        self._last_ts = ts
        rows = [
            {
                "ts": ts,
                "ticker": t,
                "weight": new[t],
                "signal_id": signal.signal_id,
                "sentiment": signal.sentiment,
                "event_class": signal.event_class.value,
                "impact": signal.impact,
            }
            for t in self.tickers
        ]
        self.store.add_weight_snapshot(rows)

        return {
            "signal_id": signal.signal_id,
            "ticker": ticker,
            "sentiment": signal.sentiment,
            "event_class": signal.event_class.value,
            "impact": signal.impact,
            "event_at": ts.isoformat(),
            "old_weight": old[ticker],
            "new_weight": new[ticker],
            "delta": new[ticker] - old[ticker],
            "turnover": sum(abs(new[t] - old[t]) for t in self.tickers) / 2.0,
            "weights": dict(new),
        }

    # ------------------------------------------------------------------ views
    def deviations(self) -> dict[str, float]:
        """Signed deviation of each weight from the equal-weight baseline."""
        return {t: self.weights[t] - self.equal_weight for t in self.tickers}

    def last_snapshot(self) -> list[dict]:
        times = self.store.snapshot_times()
        if not times:
            return []
        return [r for r in self.store.get_weight_history() if r["ts"] == times[-1]]
