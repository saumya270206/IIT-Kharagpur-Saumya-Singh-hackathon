"""The mock stock index traded by Module A (15 S&P 100 constituents with tweets + prices in the Kaggle data)."""

from __future__ import annotations

from risk_engine.companies import COMPANIES

INDEX_NAME = "Mock S&P 100 Tactical Index"

# 15 S&P 100 names that exist in BOTH Kaggle files (stock_tweets.csv + stock_yfinance_data.csv)
INDEX_TICKERS: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOG", "META",
    "TSLA", "NFLX", "AMD", "INTC", "PYPL",
    "DIS", "BA", "COST", "PG", "KO",
]


def equal_weights() -> dict[str, float]:
    w = 1.0 / len(INDEX_TICKERS)
    return {t: w for t in INDEX_TICKERS}


def constituents() -> list[dict]:
    return [
        {"ticker": t, "name": COMPANIES[t]["name"], "aliases": COMPANIES[t]["aliases"]}
        for t in INDEX_TICKERS
    ]
