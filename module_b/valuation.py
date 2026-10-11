"""Position-level revaluation under a shock (sensitivity / Greeks based).

Simplified but defensible full-portfolio model:

  Loan        rates (floating, 0.25y)  + spread-duration MtM on the fair-value
              share of the book + expected-loss uplift (EAD x LGD x PD x (PD mult - 1))
  Bond        duration + convexity     + spread-duration MtM (+ FX)
  IRS         fixed-leg duration (receive-fixed gains when rates fall) + CVA
  FX Forward  notional x FX move (+ CVA)
  Equity opt  delta-gamma (underlying move) + vega (implied vol) (+ CVA if long)
  Equity      beta x broad move + sector overlay + idiosyncratic

CVA = -EE x spread x 0.5 x min(T, 5)   (credit-triangle approximation, where
EE = positive MtM + a potential-future-exposure add-on).

Losses on cash instruments are floored at -market value (you cannot lose more
than you hold); derivatives are unbounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from module_b.portfolio import (
    LOAN_FAIR_VALUE_SHARE, LOAN_LGD, RATING_SPREAD_MULT, Position, fx_group,
)
from module_b.scenarios import Scenario

FACTORS = ("rates", "spread", "credit", "equity", "fx", "vol", "cva")

# option sensitivities (per unit of underlying notional)
_DELTA = {"Equity Call": 0.55, "Equity Put": -0.40}
_GAMMA = 3.0      # d(delta)/d(return)
_VEGA = 0.0035    # value change per vol point per unit notional


@dataclass(frozen=True)
class PositionShock:
    """The shock actually felt by one position (market + idiosyncratic)."""

    rates_bp: float
    spread_bp: float
    equity_ret: float
    fx_ret: float
    vol_pts: float
    pd_mult: float


def position_shock(pos: Position, sc: Scenario) -> PositionShock:
    m, idio = sc.market, sc.idio

    # idiosyncratic layer: full for the named counterparty, partial for its sector
    idio_k = 0.0
    if sc.ticker:
        if pos.ticker == sc.ticker:
            idio_k = 1.0
        elif sc.sector and pos.sector == sc.sector:
            idio_k = idio.sector_contagion

    rating_mult = RATING_SPREAD_MULT.get(pos.rating, 1.0)
    spread = (m.spread_bp * rating_mult
              + m.sector_spread_bp.get(pos.sector, 0.0)
              + idio.spread_bp * idio_k)
    spread = max(spread, -50.0)  # spreads cannot compress without limit

    equity = (pos.beta * m.equity_pct
              + m.sector_equity.get(pos.sector, 0.0)
              + idio.equity_pct * idio_k)
    equity = max(equity, -0.95)

    grp = fx_group(pos.currency)
    fx = 0.0 if grp == "USD" else (m.fx_major_pct if grp == "major" else m.fx_em_pct)

    pd_mult = m.pd_mult * (1.0 + (idio.pd_mult - 1.0) * idio_k)
    return PositionShock(m.rates_bp, spread, equity, fx, m.vol_pts, pd_mult)


def revalue(pos: Position, ps: PositionShock) -> dict[str, float]:
    """Return P&L by risk factor for one position (USD, negative = loss)."""
    c = {f: 0.0 for f in FACTORS}
    mv, n, d = pos.market_value, pos.notional, pos.direction
    dy = ps.rates_bp / 1e4
    ds = ps.spread_bp / 1e4

    if pos.asset_type == "Loan":
        fv = LOAN_FAIR_VALUE_SHARE
        c["rates"] = -mv * fv * pos.rate_duration * dy
        c["spread"] = -mv * fv * pos.spread_duration * ds
        c["credit"] = -n * LOAN_LGD * pos.pd * (ps.pd_mult - 1.0)
        c["fx"] = mv * ps.fx_ret

    elif pos.asset_type == "Bond":
        D, C = pos.rate_duration, pos.convexity
        c["rates"] = mv * (-D * dy + 0.5 * C * dy * dy)
        c["spread"] = -mv * pos.spread_duration * ds
        c["fx"] = mv * ps.fx_ret

    elif pos.asset_type == "Derivative":
        inst = pos.instrument
        if inst == "IRS":
            D = pos.rate_duration
            c["rates"] = -d * n * (D * dy - 0.5 * pos.convexity * dy * dy)
        elif inst == "FX Forward":
            c["fx"] = d * n * ps.fx_ret
        elif inst in _DELTA:
            r = ps.equity_ret
            c["equity"] = d * n * (_DELTA[inst] * r + 0.5 * _GAMMA * r * r)
            c["vol"] = d * n * _VEGA * ps.vol_pts
        # counterparty credit valuation adjustment on positive exposure
        held_asset = d > 0 or inst in ("IRS", "FX Forward")
        if held_asset:
            pfe = 0.015 * n * math.sqrt(max(pos.maturity_years, 0.25))
            ee = max(mv + sum(c.values()), 0.0) + pfe
            c["cva"] = -ee * ds * 0.5 * min(max(pos.maturity_years, 0.25), 5.0)

    elif pos.asset_type == "Equity":
        c["equity"] = mv * ps.equity_ret
        c["fx"] = mv * ps.fx_ret

    # cash instruments cannot lose more than their carrying value
    if pos.asset_type in ("Loan", "Bond", "Equity"):
        total = sum(c.values())
        floor = -max(mv, 0.0)
        if total < floor and total != 0:
            k = floor / total
            c = {f: v * k for f, v in c.items()}
    return c
