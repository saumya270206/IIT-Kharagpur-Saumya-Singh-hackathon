"""Turn the Kaggle "Financial Transactions" dataset into Module B's loan book.

    python scripts/prepare_transactions.py [raw_dir]

Reads (default raw_dir = data/raw/transactions):
    transactions_data.csv  13M card transactions (streamed in chunks)
    users_data.csv         client credit score, income, debt
    cards_data.csv         card credit limits
    mcc_codes.json         merchant category code -> description

Method: every client's total card credit limit (cards_data) is allocated to
(sector, US region) segments in proportion to where that client spends
(transactions_data). A segment's rating comes from the relative rank of its
spend-weighted client credit score (users_data) and its transaction error rate. Segment
exposure is scaled to wholesale size (``--book-size``, default $3.3bn).

Output: data/processed/loan_book.csv (one row per segment = one loan position).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# merchant-category description keywords -> sector (first match wins)
SECTOR_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Energy", ("service station", "fuel", "gas")),
    ("Utilities", ("utilities", "electric")),
    ("Healthcare", ("doctor", "hospital", "pharmac", "drug", "dent", "medical", "optic", "chiropract", "health")),
    ("Consumer Staples", ("grocery", "supermarket", "food", "bakeries", "candy", "liquor", "wholesale club", "discount store")),
    ("Communication Services", ("telecom", "cable", "telephone", "cellular", "satellite", "news", "publishing", "motion picture", "video", "radio", "television", "tv")),
    ("Financials", ("financial", "insurance", "money", "bank", "brokerage", "security")),
    ("Industrials", ("airline", "railroad", "bus lines", "freight", "truck", "courier", "taxi", "transportation", "tolls", "car rental", "steel", "machinery", "equipment", "contractor", "service", "repair", "freight")),
    ("Technology", ("computer", "software", "electronic", "network", "data processing")),
    ("Materials", ("lumber", "building", "hardware", "paint", "metal", "chemical")),
]
DEFAULT_SECTOR = "Consumer Discretionary"   # restaurants, retail, leisure, lodging

REGION_BY_STATE = {}
for region, states in {
    "US Northeast": "CT ME MA NH RI VT NJ NY PA",
    "US Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "US South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "US West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}.items():
    for s in states.split():
        REGION_BY_STATE[s] = region


def sector_of(desc: str) -> str:
    d = (desc or "").lower()
    for sector, keys in SECTOR_RULES:
        if any(k in d for k in keys):
            return sector
    return DEFAULT_SECTOR


def money(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def assign_ratings(avg_score: pd.Series, err_rate: pd.Series) -> pd.Series:
    """Relative internal rating. Segment averages are compressed (all ~720), so ratings
    come from the segment's rank on a risk score = low credit score + high error rate:
    best 20% A, next 40% BBB, next 30% BB, worst 10% B."""
    z = lambda x: (x - x.mean()) / (x.std() or 1.0)
    risk = z(-avg_score) + z(err_rate)
    pct = risk.rank(pct=True, method="first")
    return pd.Series(np.select([pct <= 0.2, pct <= 0.6, pct <= 0.9], ["A", "BBB", "BB"], "B"),
                     index=avg_score.index)


def build(raw: Path, book_size: float, chunksize: int = 1_000_000) -> pd.DataFrame:
    mcc = {int(k): v for k, v in json.load(open(raw / "mcc_codes.json")).items()}
    sector_by_mcc = {k: sector_of(v) for k, v in mcc.items()}
    users = pd.read_csv(raw / "users_data.csv")
    cards = pd.read_csv(raw / "cards_data.csv")
    users["credit_score"] = pd.to_numeric(users["credit_score"], errors="coerce")
    limit = cards.assign(limit=money(cards["credit_limit"])).groupby("client_id")["limit"].sum()

    parts = []
    for i, ch in enumerate(pd.read_csv(
            raw / "transactions_data.csv", chunksize=chunksize,
            usecols=["client_id", "amount", "merchant_state", "mcc", "errors"],
            dtype={"merchant_state": "string", "errors": "string"})):
        ch["amt"] = money(ch["amount"])
        ch = ch[ch["amt"] > 0]                        # spend only, not refunds
        ch["sector"] = ch["mcc"].map(sector_by_mcc).fillna(DEFAULT_SECTOR)
        ch["region"] = ch["merchant_state"].map(REGION_BY_STATE).fillna("US Online/Other")
        ch["err"] = ch["errors"].notna().astype(int)
        parts.append(ch.groupby(["client_id", "sector", "region"])
                       .agg(spend=("amt", "sum"), n=("amt", "size"), err=("err", "sum"))
                       .reset_index())
        print(f"  chunk {i + 1} done", file=sys.stderr)
    g = (pd.concat(parts).groupby(["client_id", "sector", "region"], as_index=False)
           .sum(numeric_only=True))

    g["share"] = g["spend"] / g.groupby("client_id")["spend"].transform("sum")
    g["limit_alloc"] = g["client_id"].map(limit).fillna(0.0) * g["share"]
    g = g.merge(users[["id", "credit_score"]].rename(columns={"id": "client_id"}),
                on="client_id", how="left")
    g["score_w"] = g["credit_score"].fillna(g["credit_score"].median()) * g["limit_alloc"]

    seg = g.groupby(["sector", "region"], as_index=False).agg(
        limit=("limit_alloc", "sum"), score_w=("score_w", "sum"),
        spend=("spend", "sum"), n=("n", "sum"), err=("err", "sum"),
        clients=("client_id", "nunique"))
    seg = seg[seg["limit"] > 0].copy()
    seg["avg_score"] = seg["score_w"] / seg["limit"]
    seg["err_rate"] = seg["err"] / seg["n"]
    seg["rating"] = assign_ratings(seg["avg_score"], seg["err_rate"])

    scale = book_size / seg["limit"].sum()
    seg["notional"] = (seg["limit"] * scale).round(0)
    disc = {"AA": 0.002, "A": 0.004, "BBB": 0.008, "BB": 0.02, "B": 0.04, "CCC": 0.12}
    seg["market_value"] = (seg["notional"] * (1 - seg["rating"].map(disc))).round(2)
    rng = np.random.default_rng(7)                       # deterministic tenor / coupon
    seg["maturity_years"] = np.round(rng.uniform(1, 7, len(seg)), 2)
    cpn = {"AA": 4.5, "A": 5.0, "BBB": 5.8, "BB": 6.8, "B": 8.0, "CCC": 10.0}
    seg["coupon_pct"] = seg["rating"].map(cpn)
    seg = seg.sort_values("notional", ascending=False).reset_index(drop=True)
    return pd.DataFrame({
        "trade_id": [f"LN-{i + 1:04d}" for i in range(len(seg))],
        "counterparty": [f"{r.sector} merchants - {r.region}" for r in seg.itertuples()],
        "ticker": "", "asset_type": "Loan",
        "instrument": np.where(seg["err_rate"] > 0.0, "Revolver", "Term Loan"),
        "sector": seg["sector"], "region": seg["region"], "currency": "USD",
        "rating": seg["rating"], "notional": seg["notional"],
        "market_value": seg["market_value"], "maturity_years": seg["maturity_years"],
        "coupon_pct": seg["coupon_pct"], "direction": 1,
        "clients": seg["clients"], "avg_credit_score": seg["avg_score"].round(0),
        "txn_error_rate": seg["err_rate"].round(5),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", default=str(ROOT / "data/raw/transactions"))
    ap.add_argument("--book-size", type=float, default=3.3e9)
    ap.add_argument("--out", default=str(ROOT / "data/processed/loan_book.csv"))
    a = ap.parse_args()
    df = build(Path(a.raw_dir), a.book_size)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} loan segments, ${df.notional.sum() / 1e9:.2f}bn -> {a.out}")
