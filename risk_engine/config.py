"""Central configuration for the risk engine (paths, env vars, sources)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("RISK_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = Path(os.getenv("RISK_RAW_DIR", DATA_DIR / "raw"))            # Kaggle datasets go here
PROCESSED_DIR = Path(os.getenv("RISK_PROCESSED_DIR", DATA_DIR / "processed"))
OUTPUT_DIR = Path(os.getenv("RISK_OUTPUT_DIR", DATA_DIR / "output"))
DB_PATH = Path(os.getenv("RISK_DB_PATH", OUTPUT_DIR / "risk.db"))
SIGNALS_JSONL = Path(os.getenv("RISK_SIGNALS_JSONL", OUTPUT_DIR / "signals.jsonl"))

# Kaggle datasets (see README "Datasets"). Replay size is capped so a full run stays fast.
TWEETS_CSV = Path(os.getenv("TWEETS_CSV", RAW_DIR / "stock_tweets.csv"))
PRICES_CSV = Path(os.getenv("PRICES_CSV", RAW_DIR / "stock_yfinance_data.csv"))
PHRASEBANK_TXT = Path(os.getenv("PHRASEBANK_TXT", RAW_DIR / "FinancialPhraseBank" / "Sentences_AllAgree.txt"))
PHRASEBANK_ALL = Path(os.getenv("PHRASEBANK_ALL", RAW_DIR / "all-data.csv"))
TRANSACTIONS_DIR = Path(os.getenv("TRANSACTIONS_DIR", RAW_DIR / "transactions"))
DATASET_TWEET_LIMIT = int(os.getenv("DATASET_TWEET_LIMIT", "3000"))   # 0 = all
DATASET_NEWS_LIMIT = int(os.getenv("DATASET_NEWS_LIMIT", "400"))      # 0 = all

API_HOST = os.getenv("RISK_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("RISK_API_PORT", "8000"))

# NLP backend selection: auto | off | local | openai | anthropic
# "auto" uses an LLM when an API key is present, otherwise the local lexicon model.
ENGINE_LLM = os.getenv("ENGINE_LLM", "auto").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# ---------------------------------------------------------------------------
# Live news feeds (all free; only NewsAPI needs a key). Every URL can be
# overridden by env var, e.g. to point at a corporate proxy or a test server.
# ---------------------------------------------------------------------------
_COMPANIES = ('(Apple OR Microsoft OR Nvidia OR Amazon OR Google OR Meta OR Tesla OR '
              'JPMorgan OR "Bank of America" OR Walmart OR Exxon OR "Johnson & Johnson" '
              'OR "Procter & Gamble" OR UnitedHealth OR AMD)')

# GDELT DOC 2.0 (keyless). Etiquette: max 1 request / 5 s.
GDELT_URL = os.getenv("GDELT_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
# GDELT rejects long queries ("Your query was too short or too long."), so the default is
# several short queries - one per batch of index members - merged and de-duplicated.
# Set GDELT_QUERY to force one custom query instead.
_GDELT_BATCHES = (
    '(Apple OR Microsoft OR Amazon OR Google OR Meta)',
    '(Tesla OR Netflix OR AMD OR Intel OR PayPal)',
    '(Disney OR Boeing OR Costco OR Procter OR "Coca-Cola")',
)
_GDELT_TOPICS = "(earnings OR shares OR stock OR lawsuit OR downgrade OR probe) sourcelang:english"
GDELT_QUERIES: list[str] = (
    [os.environ["GDELT_QUERY"]] if os.getenv("GDELT_QUERY")
    else [f"{b} {_GDELT_TOPICS}" for b in _GDELT_BATCHES]
)
GDELT_QUERY = GDELT_QUERIES[0]                                   # kept for backward compatibility
GDELT_BATCH_PAUSE = float(os.getenv("GDELT_BATCH_PAUSE", "5.5"))  # GDELT etiquette: 1 request / 5 s
GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "25"))
GDELT_TIMEOUT = float(os.getenv("GDELT_TIMEOUT", "20"))
GDELT_TIMESPAN = os.getenv("GDELT_TIMESPAN", "24h")

# Google News RSS (keyless)
GOOGLE_NEWS_URL = os.getenv("GOOGLE_NEWS_URL", "https://news.google.com/rss/search")
RSS_QUERY = os.getenv("RSS_QUERY", _COMPANIES + " (stock OR earnings OR shares) when:2d")
RSS_TIMEOUT = float(os.getenv("RSS_TIMEOUT", "15"))

# Yahoo Finance per-ticker RSS (keyless)
YAHOO_RSS_URL = os.getenv("YAHOO_RSS_URL", "https://feeds.finance.yahoo.com/rss/2.0/headline")
YAHOO_TICKERS = [t.strip().upper() for t in os.getenv(
    "YAHOO_TICKERS", "AAPL,MSFT,AMZN,GOOG,META,TSLA,NFLX,AMD,INTC,PYPL,DIS,BA,COST,PG,KO"
).split(",") if t.strip()]
YAHOO_PER_TICKER = int(os.getenv("YAHOO_PER_TICKER", "3"))

# NewsAPI.org (free developer key) - only used when NEWSAPI_KEY is set
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_URL = os.getenv("NEWSAPI_URL", "https://newsapi.org/v2/everything")
NEWSAPI_QUERY = os.getenv("NEWSAPI_QUERY", _COMPANIES + " AND (stock OR earnings OR shares)")

# Shared HTTP behaviour
HTTP_RETRIES = int(os.getenv("RISK_HTTP_RETRIES", "2"))      # extra attempts after the first
HTTP_BACKOFF = float(os.getenv("RISK_HTTP_BACKOFF", "2.0"))  # seconds, doubled each retry
HTTP_USER_AGENT = os.getenv(
    "RISK_USER_AGENT",
    "Mozilla/5.0 (compatible; RiskEngine/1.0; +https://example.com/risk-engine)")
# TLS: set RISK_CA_BUNDLE=/path/to/ca.pem behind a corporate proxy, or
# RISK_SSL_VERIFY=0 as a last resort (insecure). Proxies: standard HTTPS_PROXY / HTTP_PROXY.
CA_BUNDLE = os.getenv("RISK_CA_BUNDLE", "")
SSL_VERIFY = os.getenv("RISK_SSL_VERIFY", "1").lower() not in ("0", "false", "no", "off")

for _d in (DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)