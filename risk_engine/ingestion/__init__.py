"""Ingestion package: Kaggle datasets + live feeds (GDELT, Google News, Yahoo, NewsAPI)."""

from risk_engine.ingestion.gdelt import fetch_gdelt
from risk_engine.ingestion.newsapi import fetch_newsapi
from risk_engine.ingestion.rss import fetch_rss, fetch_yahoo
from risk_engine.ingestion.datasets import load_datasets

__all__ = ["fetch_gdelt", "fetch_rss", "fetch_yahoo", "fetch_newsapi", "load_datasets"]