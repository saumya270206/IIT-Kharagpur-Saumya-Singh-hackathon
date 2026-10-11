"""Validate the NLP engine and Module A against the Kaggle datasets.

    python scripts/evaluate.py            # both checks
    python scripts/evaluate.py --skip-tweets

1. Sentiment accuracy on Financial PhraseBank (labelled positive/negative/neutral).
2. Signal quality on real tweets: does the sentiment score of a stock's tweets on
   day t-1 line up with that stock's return on day t? (Spearman rank correlation of
   the per-stock daily mean sentiment vs next-day return - i.e. an information coefficient.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_engine import config  # noqa: E402
from risk_engine.nlp.local import score_sentiment  # noqa: E402
from risk_engine.ingestion.datasets import _clean  # noqa: E402


def label_of(score: float, t: float) -> str:
    return "positive" if score > t else "negative" if score < -t else "neutral"


def prf(y, p, labels=("negative", "neutral", "positive")):
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def phrasebank_eval(t: float = 0.15) -> None:
    df = pd.read_csv(config.PHRASEBANK_ALL, header=None, names=["label", "text"],
                     encoding="latin-1")
    df["score"] = df["text"].map(score_sentiment)
    print(f"\n== Financial PhraseBank ({len(df)} sentences, neutral band +/-{t}) ==")
    df["pred"] = df["score"].map(lambda s: label_of(s, t))
    acc = float((df["pred"] == df["label"]).mean())
    print(f"accuracy {acc:.3f} | macro-F1 {prf(df['label'], df['pred']):.3f} "
          f"| majority-class baseline {df['label'].value_counts(normalize=True).max():.3f}")
    print(pd.crosstab(df["label"], df["pred"], rownames=["true"], colnames=["pred"]))
    pol = df[df["label"] != "neutral"]
    da = float(((pol["score"] > 0) == (pol["label"] == "positive")).mean())
    print(f"polarity accuracy on non-neutral sentences (direction only): {da:.3f}")
    for tt in (0.05, 0.10, 0.15, 0.25, 0.35):
        p = df["score"].map(lambda s: label_of(s, tt))
        print(f"  band +/-{tt:.2f}: acc {float((p == df['label']).mean()):.3f} "
              f"F1 {prf(df['label'], p):.3f}")


def tweet_ic(limit: int = 0) -> None:
    from module_a.index import INDEX_TICKERS
    tw = pd.read_csv(config.TWEETS_CSV)
    tw = tw[tw["Stock Name"].isin(INDEX_TICKERS)].copy()
    tw["text"] = tw["Tweet"].map(_clean)
    tw = tw[tw["text"].str.len() >= 25].drop_duplicates(["Stock Name", "text"])
    if limit:
        tw = tw.sample(min(limit, len(tw)), random_state=7)
    tw["score"] = tw["text"].map(score_sentiment)
    tw["day"] = pd.to_datetime(tw["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    daily = tw.groupby(["day", "Stock Name"])["score"].agg(["mean", "size"]).reset_index()
    px = pd.read_csv(config.PRICES_CSV, parse_dates=["Date"])
    px = px.pivot_table(index="Date", columns="Stock Name", values="Adj Close").sort_index()
    ret = px.pct_change().stack().rename("ret").reset_index()
    ret.columns = ["day", "Stock Name", "ret"]
    # tweets on day d (any time) -> return on the next trading day d+1 (no look-ahead)
    nxt = {d: n for d, n in zip(px.index[:-1], px.index[1:])}
    daily["trade_day"] = daily["day"].map(nxt)
    m = daily.dropna(subset=["trade_day"]).merge(
        ret, left_on=["trade_day", "Stock Name"], right_on=["day", "Stock Name"], how="inner")
    m = m[m["size"] >= 3]
    print(f"\n== Tweet sentiment vs next-day return ({len(tw)} tweets, {len(m)} stock-days) ==")
    pooled = m["mean"].corr(m["ret"], method="spearman")
    # per-day cross-sectional IC (rank stocks by sentiment, compare with return ranks)
    ics = [g["mean"].corr(g["ret"], method="spearman")
           for _, g in m.groupby("trade_day") if len(g) >= 5]
    ics = pd.Series(ics).dropna()
    print(f"pooled Spearman corr: {pooled:+.3f}")
    if len(ics):
        print(f"mean daily cross-sectional IC: {ics.mean():+.3f} over {len(ics)} days "
              f"(t-stat {ics.mean() / (ics.std() / np.sqrt(len(ics))):+.2f})")
    hit = ((m["mean"] > 0) == (m["ret"] > 0))[m["mean"].abs() > 0.1]
    print(f"directional hit-rate (|sentiment|>0.1): {hit.mean():.3f} on {int(hit.size)} stock-days")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tweets", action="store_true")
    ap.add_argument("--band", type=float, default=0.15)
    a = ap.parse_args()
    phrasebank_eval(a.band)
    if not a.skip_tweets:
        tweet_ic()
