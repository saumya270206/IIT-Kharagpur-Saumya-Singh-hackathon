"""Dataset ingestion: real Kaggle tweets + financial news (replaces the old samples).

Two sources, both real data:

* ``twitter`` - ``stock_tweets.csv`` (Kaggle "Tweets about the Top Companies",
  Sep-2021 .. Sep-2022). Each row already names its stock, used as the ticker hint.
* ``news``    - Financial PhraseBank (``Sentences_AllAgree.txt``), professionally
  annotated financial-news sentences. They carry no timestamp or ticker, so they are
  spread evenly over the tweet period (deterministic) and stay market-level events.

Replay is capped (``DATASET_TWEET_LIMIT`` / ``DATASET_NEWS_LIMIT``, 0 = everything)
and tweets are balanced per ticker so TSLA (46% of the file) cannot drown the rest.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from risk_engine import config
from risk_engine.models import RawItem

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
# replay window of the tweet file (also used to place the undated news sentences)
WINDOW_START = datetime(2021, 10, 1, 14, 30, tzinfo=timezone.utc)
WINDOW_END = datetime(2022, 9, 28, 20, 0, tzinfo=timezone.utc)


def _clean(text: str) -> str:
    text = html.unescape(str(text))
    return _WS.sub(" ", _URL.sub("", text)).strip()


def _hash(*parts: object) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:14]


def _need(path: Path, what: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Download the Kaggle dataset and put it under "
            f"{config.RAW_DIR} (see README 'Datasets').")
    return Path(path)


def load_tweets(path: Path | str | None = None, tickers: Optional[Sequence[str]] = None,
                limit: Optional[int] = None, seed: int = 7) -> list[RawItem]:
    from module_a.index import INDEX_TICKERS
    tickers = list(tickers or INDEX_TICKERS)
    limit = config.DATASET_TWEET_LIMIT if limit is None else limit
    df = pd.read_csv(_need(path or config.TWEETS_CSV, "stock_tweets.csv"))
    df = df[df["Stock Name"].isin(tickers)].copy()
    df["text"] = df["Tweet"].map(_clean)
    df = df[df["text"].str.len() >= 25]                       # drop link-only / empty
    df = df.drop_duplicates(subset=["Stock Name", "text"])
    df["ts"] = pd.to_datetime(df["Date"], utc=True)
    if limit and len(df) > limit:                             # balanced per-ticker cap
        cap = -(-limit // df["Stock Name"].nunique())
        df = df.sample(frac=1.0, random_state=seed)            # shuffle, then cap per ticker
        df = df[df.groupby("Stock Name").cumcount() < cap]
    df = df.sort_values("ts")
    return [RawItem(id="tw-" + _hash(ts.isoformat(), tk, tx), source="twitter", text=tx,
                    published_at=ts.to_pydatetime(), ticker=tk, company=co)
            for ts, tk, tx, co in zip(df["ts"], df["Stock Name"], df["text"], df["Company Name"])]


def load_phrasebank(path: Path | str | None = None) -> pd.DataFrame:
    """Financial PhraseBank as a DataFrame(text, label). Format: ``sentence@label``."""
    rows = []
    with open(_need(path or config.PHRASEBANK_TXT, "Financial PhraseBank"),
              encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if "@" in line:
                text, label = line.rsplit("@", 1)
                rows.append((text.strip(), label.strip().lower()))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_news(path: Path | str | None = None, limit: Optional[int] = None,
              seed: int = 7) -> list[RawItem]:
    limit = config.DATASET_NEWS_LIMIT if limit is None else limit
    df = load_phrasebank(path).drop_duplicates("text")
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=seed)
    df = df.reset_index(drop=True)
    step = (WINDOW_END - WINDOW_START) / max(len(df), 1)
    return [RawItem(id="news-" + _hash(r.text), source="news", text=r.text,
                    published_at=WINDOW_START + step * i)
            for i, r in enumerate(df.itertuples())]


def load_datasets() -> list[RawItem]:
    """News + tweets, chronologically ordered (the order signals are processed)."""
    items = [*load_news(), *load_tweets()]
    return sorted(items, key=lambda i: i.published_at)
