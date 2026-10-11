"""Wholesale-banking portfolio (loans, bonds, derivatives, equities).

* **Loans** come from the real Kaggle "Financial Transactions" dataset: 13M card
  transactions + client credit limits/scores aggregated into sector x region credit
  segments by ``scripts/prepare_transactions.py`` -> ``data/processed/loan_book.csv``.
* **Bonds, derivatives and equities** are not in that dataset (it is card data), so a
  small deterministic synthetic trading book is generated for them.

Any other transaction table can be mapped in via :func:`from_transactions_df`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from risk_engine import config

LOAN_BOOK_PATH = Path(config.PROCESSED_DIR) / "loan_book.csv"

ASSET_TYPES = ("Loan", "Bond", "Derivative", "Equity")
INSTRUMENTS = {
    "Loan": ("Term Loan", "Revolver"),
    "Bond": ("Corporate Bond",),
    "Derivative": ("IRS", "FX Forward", "Equity Call", "Equity Put"),
    "Equity": ("Equity",),
}

# --- reference data ---------------------------------------------------------
SECTOR_BY_TICKER: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "AMD": "Technology",
    "INTC": "Technology", "TSM": "Technology", "NVDA": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "GOOG": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "JPM": "Financials", "BAC": "Financials", "PYPL": "Financials",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "COST": "Consumer Staples",
    "BA": "Industrials", "XOM": "Energy",
}

SECTOR_BETA: dict[str, float] = {
    "Technology": 1.25, "Financials": 1.15, "Consumer Discretionary": 1.20,
    "Communication Services": 1.10, "Industrials": 1.00, "Materials": 1.05,
    "Energy": 0.95, "Healthcare": 0.70, "Consumer Staples": 0.60, "Utilities": 0.50,
}

# one-year probability of default by rating (through-the-cycle, rounded)
RATING_PD: dict[str, float] = {
    "AAA": 0.0001, "AA": 0.0003, "A": 0.0008, "BBB": 0.0020,
    "BB": 0.0100, "B": 0.0400, "CCC": 0.2000,
}
# how much of a market spread move a rating bucket experiences (BBB = 1.0)
RATING_SPREAD_MULT: dict[str, float] = {
    "AAA": 0.35, "AA": 0.50, "A": 0.70, "BBB": 1.00,
    "BB": 1.80, "B": 2.60, "CCC": 3.50,
}
LOAN_LGD = 0.45  # senior unsecured (Basel foundation-IRB)
# Share of the loan book carried at fair value (held-for-sale / syndication
# pipeline). The rest is held at amortised cost, so it only feels expected-loss
# (PD) deterioration, not spread mark-to-market.
LOAN_FAIR_VALUE_SHARE = 0.5

MAJOR_CCY = {"EUR", "GBP", "JPY", "CHF", "CAD", "AUD"}


def fx_group(currency: str) -> str:
    cur = (currency or "USD").upper()
    if cur == "USD":
        return "USD"
    return "major" if cur in MAJOR_CCY else "em"


def _par_mod_duration(years: float, yld: float) -> float:
    """Modified duration of a par bond/swap fixed leg."""
    years = max(float(years), 0.05)
    yld = max(float(yld), 0.005)
    return (1.0 - (1.0 + yld) ** -years) / yld


# --- position -----------------------------------------------------------------
@dataclass
class Position:
    trade_id: str
    asset_type: str          # Loan | Bond | Derivative | Equity
    instrument: str
    counterparty: str
    sector: str
    region: str
    currency: str
    rating: str
    notional: float          # USD
    market_value: float      # USD, signed (derivative MtM can be negative)
    maturity_years: float
    coupon_pct: float = 0.0
    direction: int = 1       # +1 long / receive-fixed / buy ; -1 short / pay-fixed / sell
    ticker: Optional[str] = None

    # ---- derived risk sensitivities ----
    @property
    def yield_proxy(self) -> float:
        return max(self.coupon_pct / 100.0, 0.02)

    @property
    def rate_duration(self) -> float:
        """Modified duration to a parallel rate move."""
        if self.asset_type == "Loan":
            return 0.25  # floating-rate: reprices quarterly
        if self.asset_type in ("Bond", "Derivative"):
            return _par_mod_duration(self.maturity_years, self.yield_proxy)
        return 0.0

    @property
    def convexity(self) -> float:
        d = self.rate_duration
        return d * (d + 1.0) * 0.9

    @property
    def spread_duration(self) -> float:
        if self.asset_type == "Loan":
            return min(self.maturity_years, 5.0) * 0.9
        if self.asset_type == "Bond":
            return _par_mod_duration(self.maturity_years, self.yield_proxy)
        return 0.0

    @property
    def beta(self) -> float:
        return SECTOR_BETA.get(self.sector, 1.0)

    @property
    def pd(self) -> float:
        return RATING_PD.get(self.rating, RATING_PD["BBB"])

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id, "asset_type": self.asset_type,
            "instrument": self.instrument, "counterparty": self.counterparty,
            "ticker": self.ticker, "sector": self.sector, "region": self.region,
            "currency": self.currency, "rating": self.rating,
            "notional": self.notional, "market_value": self.market_value,
            "maturity_years": self.maturity_years, "coupon_pct": self.coupon_pct,
            "direction": self.direction,
        }


@dataclass
class Portfolio:
    positions: list[Position] = field(default_factory=list)
    capital_ratio: float = 0.06   # loss-absorbing capital as share of gross exposure
    name: str = "Wholesale Banking Book (Kaggle card-credit loans + synthetic trading book)"

    @property
    def value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions)

    @property
    def capital_buffer(self) -> float:
        return self.capital_ratio * self.gross_exposure

    def tickers(self) -> set[str]:
        return {p.ticker for p in self.positions if p.ticker}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.positions])

    def summary(self) -> dict:
        df = self.to_frame()
        return {
            "name": self.name,
            "positions": len(self.positions),
            "net_value": self.value,
            "gross_exposure": self.gross_exposure,
            "capital_buffer": self.capital_buffer,
            "by_asset_type": df.groupby("asset_type")["market_value"].sum().to_dict(),
            "by_sector": df.groupby("sector")["market_value"].sum().to_dict(),
        }


# --- synthetic transaction generator -----------------------------------------
# (counterparty, ticker, sector, region, rating, currency)
_NAMES = [
    ("Apple", "AAPL", "Technology", "North America", "AA", "USD"),
    ("Microsoft", "MSFT", "Technology", "North America", "AAA", "USD"),
    ("Amazon", "AMZN", "Consumer Discretionary", "North America", "AA", "USD"),
    ("Alphabet", "GOOG", "Communication Services", "North America", "AA", "USD"),
    ("Meta Platforms", "META", "Communication Services", "North America", "AA", "USD"),
    ("Tesla", "TSLA", "Consumer Discretionary", "North America", "BBB", "USD"),
    ("Netflix", "NFLX", "Communication Services", "North America", "BBB", "USD"),
    ("Advanced Micro Devices", "AMD", "Technology", "North America", "BBB", "USD"),
    ("Intel", "INTC", "Technology", "North America", "A", "USD"),
    ("PayPal", "PYPL", "Financials", "North America", "A", "USD"),
    ("Walt Disney", "DIS", "Communication Services", "North America", "A", "USD"),
    ("Boeing", "BA", "Industrials", "North America", "BBB", "USD"),
    ("Costco", "COST", "Consumer Staples", "North America", "AA", "USD"),
    ("Procter & Gamble", "PG", "Consumer Staples", "North America", "AA", "USD"),
    ("Coca-Cola", "KO", "Consumer Staples", "North America", "A", "USD"),
    ("JPMorgan Chase", "JPM", "Financials", "North America", "A", "USD"),
    ("Bank of America", "BAC", "Financials", "North America", "A", "USD"),
    ("Exxon Mobil", "XOM", "Energy", "North America", "AA", "USD"),
    ("Meridian Shipping Ltd", None, "Industrials", "Europe", "BB", "EUR"),
    ("Andes Mining Corp", None, "Materials", "Latin America", "B", "BRL"),
    ("Sakura Industrial KK", None, "Industrials", "Asia Pacific", "A", "JPY"),
    ("Gulf Petrochem", None, "Energy", "Middle East", "BBB", "USD"),
    ("Bharat Infra Ltd", None, "Industrials", "Asia Pacific", "BBB", "INR"),
    ("Nordic Wind AS", None, "Utilities", "Europe", "BBB", "EUR"),
    ("Thames Water Utilities", None, "Utilities", "Europe", "BB", "GBP"),
]


def generate_transactions(seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic TRADING book: bonds, derivatives, equities (no loans -
    the loan book comes from the Kaggle transactions, see ``load_loan_book``)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n = 0

    def tid(prefix: str) -> str:
        nonlocal n
        n += 1
        return f"{prefix}-{n:04d}"

    def pick():
        return _NAMES[int(rng.integers(len(_NAMES)))]

    # Bonds: held-for-trading corporate bonds
    for _ in range(16):
        cp, tk, sec, reg, rat, ccy = pick()
        notional = float(rng.integers(10, 120)) * 1e6
        coupon = round(float(rng.uniform(2.5, 6.5)), 2)
        price = float(rng.uniform(0.94, 1.04))
        rows.append(dict(
            trade_id=tid("BD"), counterparty=cp, ticker=tk, asset_type="Bond",
            instrument="Corporate Bond", sector=sec, region=reg, currency=ccy,
            rating=rat, notional=notional, market_value=round(notional * price, 2),
            maturity_years=round(float(rng.uniform(2, 15)), 2),
            coupon_pct=coupon, direction=1))

    # Derivatives: IRS, FX forwards, equity options
    for _ in range(7):  # interest-rate swaps
        cp, tk, sec, reg, rat, ccy = pick()
        notional = float(rng.integers(50, 400)) * 1e6
        direction = int(rng.choice([1, -1]))  # +1 receive-fixed, -1 pay-fixed
        rows.append(dict(
            trade_id=tid("DV"), counterparty=cp, ticker=tk, asset_type="Derivative",
            instrument="IRS", sector=sec, region=reg, currency="USD", rating=rat,
            notional=notional, market_value=round(notional * rng.uniform(-0.012, 0.02), 2),
            maturity_years=round(float(rng.uniform(2, 10)), 2),
            coupon_pct=round(float(rng.uniform(2.5, 4.5)), 2), direction=direction))
    non_usd = [x for x in _NAMES if x[5] != "USD"]
    for _ in range(5):  # FX forwards (non-USD)
        cp, tk, sec, reg, rat, ccy = non_usd[int(rng.integers(len(non_usd)))]
        notional = float(rng.integers(20, 150)) * 1e6
        rows.append(dict(
            trade_id=tid("DV"), counterparty=cp, ticker=tk, asset_type="Derivative",
            instrument="FX Forward", sector=sec, region=reg, currency=ccy,
            rating=rat, notional=notional,
            market_value=round(notional * rng.uniform(-0.01, 0.01), 2),
            maturity_years=round(float(rng.uniform(0.25, 2)), 2), coupon_pct=0.0,
            direction=int(rng.choice([1, -1]))))
    index_names = [x for x in _NAMES if x[1] is not None]
    for kind in ("Equity Call", "Equity Put", "Equity Put", "Equity Call", "Equity Put"):
        cp, tk, sec, reg, rat, ccy = index_names[int(rng.integers(len(index_names)))]
        notional = float(rng.integers(10, 80)) * 1e6
        rows.append(dict(
            trade_id=tid("DV"), counterparty=cp, ticker=tk, asset_type="Derivative",
            instrument=kind, sector=sec, region=reg, currency="USD", rating=rat,
            notional=notional, market_value=round(notional * rng.uniform(0.03, 0.08), 2),
            maturity_years=round(float(rng.uniform(0.5, 2)), 2), coupon_pct=0.0,
            direction=1))

    # Equities: trading-book positions in index names
    for cp, tk, sec, reg, rat, ccy in [x for x in _NAMES if x[1]]:
        value = float(rng.integers(8, 70)) * 1e6
        rows.append(dict(
            trade_id=tid("EQ"), counterparty=cp, ticker=tk, asset_type="Equity",
            instrument="Equity", sector=sec, region=reg, currency="USD", rating=rat,
            notional=value, market_value=value, maturity_years=0.0,
            coupon_pct=0.0, direction=1))

    return pd.DataFrame(rows)


# --- loading ---------------------------------------------------------------
_ALIASES = {
    "trade_id": ["trade_id", "transaction_id", "id", "txn_id"],
    "counterparty": ["counterparty", "merchant", "client", "customer", "name"],
    "asset_type": ["asset_type", "asset_class", "type", "product"],
    "instrument": ["instrument", "product_type", "mcc_description"],
    "sector": ["sector", "industry", "category"],
    "region": ["region", "country", "merchant_state"],
    "currency": ["currency", "ccy"],
    "rating": ["rating", "credit_rating"],
    "notional": ["notional", "amount", "exposure", "principal"],
    "market_value": ["market_value", "mtm", "value", "mv"],
    "maturity_years": ["maturity_years", "tenor", "tenor_years", "maturity"],
    "coupon_pct": ["coupon_pct", "coupon", "rate", "interest_rate"],
    "direction": ["direction", "side"],
    "ticker": ["ticker", "symbol"],
}


def from_transactions_df(df: pd.DataFrame, **kwargs) -> Portfolio:
    """Build a Portfolio from any transaction table, tolerating missing columns
    (sensible defaults) and common column-name aliases."""
    cols = {c.lower().strip(): c for c in df.columns}

    def col(key: str):
        for alias in _ALIASES[key]:
            if alias in cols:
                return df[cols[alias]]
        return None

    n = len(df)
    notional = pd.to_numeric(col("notional"), errors="coerce") if col("notional") is not None \
        else pd.Series([1e7] * n)
    notional = notional.abs().fillna(1e7)
    mv = pd.to_numeric(col("market_value"), errors="coerce") if col("market_value") is not None \
        else notional
    mv = mv.fillna(notional)

    def text(key: str, default: str) -> pd.Series:
        s = col(key)
        return (s.astype(str).fillna(default) if s is not None
                else pd.Series([default] * n)).reset_index(drop=True)

    def num(key: str, default: float) -> pd.Series:
        s = col(key)
        return (pd.to_numeric(s, errors="coerce").fillna(default) if s is not None
                else pd.Series([default] * n)).reset_index(drop=True)

    asset = text("asset_type", "Loan").str.title()
    asset = asset.where(asset.isin(ASSET_TYPES), "Loan")
    ticker = col("ticker")
    ticker = (ticker.astype("string") if ticker is not None
              else pd.Series([None] * n, dtype="string")).reset_index(drop=True)

    positions = []
    ids, cps, inst = text("trade_id", ""), text("counterparty", "Unknown"), text("instrument", "")
    sec, reg = text("sector", "Industrials"), text("region", "North America")
    cur, rat = text("currency", "USD"), text("rating", "BBB")
    mat, cpn = num("maturity_years", 3.0), num("coupon_pct", 4.0)
    side = col("direction")
    notional, mv = notional.reset_index(drop=True), mv.reset_index(drop=True)
    for i in range(n):
        a = asset.iloc[i]
        direction = 1
        if side is not None:
            v = str(side.iloc[i]).lower()
            direction = -1 if v in ("-1", "sell", "short", "pay", "payer") else 1
        t = ticker.iloc[i]
        positions.append(Position(
            trade_id=ids.iloc[i] or f"TX-{i + 1:05d}", asset_type=a,
            instrument=inst.iloc[i] or INSTRUMENTS[a][0],
            counterparty=cps.iloc[i], sector=sec.iloc[i], region=reg.iloc[i],
            currency=cur.iloc[i].upper(), rating=rat.iloc[i].upper(),
            notional=float(notional.iloc[i]), market_value=float(mv.iloc[i]),
            maturity_years=float(mat.iloc[i]), coupon_pct=float(cpn.iloc[i]),
            direction=direction,
            ticker=None if pd.isna(t) else str(t).upper()))
    return Portfolio(positions=positions, **kwargs)


def load_loan_book(path: Path | str | None = None) -> pd.DataFrame:
    """Loan positions derived from the Kaggle transactions (see scripts/prepare_transactions.py)."""
    p = Path(path) if path else LOAN_BOOK_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it from the Kaggle 'Financial Transactions' data with:\n"
            f"    python scripts/prepare_transactions.py {config.TRANSACTIONS_DIR}")
    return pd.read_csv(p)


def load_portfolio(path: Path | str | None = None, **kwargs) -> Portfolio:
    """Default book = Kaggle-derived loans + synthetic bonds/derivatives/equities.
    Pass ``path`` to load any other transaction CSV instead."""
    if path:
        return from_transactions_df(pd.read_csv(path), **kwargs)
    df = pd.concat([load_loan_book(), generate_transactions()], ignore_index=True)
    return from_transactions_df(df, **kwargs)


if __name__ == "__main__":  # python -m module_b.portfolio -> print a summary
    pf = load_portfolio()
    print(f"{len(pf.positions)} positions, net ${pf.value / 1e9:.2f}bn")
    print(pf.to_frame().groupby("asset_type")["market_value"].agg(["count", "sum"]))
