# AI/NLP Risk Engine for Tactical Index Rebalancing & Portfolio Stress Testing - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Saumya Singh
**College Email ID:** saumya2024.24@kgpian.iitkgp.ac.in
**College / Campus:** Indian Institute of Technology, Kharagpur
**Demo Video Link:** [YouTube unlisted link]
**Slide Deck Link (if hosted externally):** Not hosted externally - see [`docs/presentation.pdf`](docs/presentation.pdf)

---

## 1. Project Overview / Problem Statement & Approach

Market-moving information (news headlines, social media posts) arrives as unstructured text, far faster than
analysts can read it. The problem: **turn that text into structured, machine-usable risk signals in near real time,
and let two very different downstream users act on the same signals** - a tactical index desk that rebalances in
minutes, and a strategic risk team that needs to know when a bank's loan book is exposed.

**Approach.** A single *AI/NLP Risk Engine* ingests news (GDELT, Google News RSS, Yahoo Finance RSS, optional
NewsAPI, plus bundled Kaggle replays) and tweets. For every item it produces a structured signal:
**Sentiment Score (-1 to +1)**, **Event Class** (e.g. Credit Event), **Impact Score (1-10)**, plus ticker and
confidence. The signal stream is persisted (SQLite + JSONL), exposed through a FastAPI service and *published* to
subscribers. The engine never imports its consumers, so modules plug in with `pipeline.subscribe(handler)` and a
failing subscriber cannot stop the feed.

* **Module A - Tactical Index Rebalancing.** Each signal nudges the weight of a mock S&P 100 index (15
  constituents) up or down, scaled by impact, under weight bounds and a turnover cap so the index stays
  investable. A back-test on real prices compares the sentiment-driven index to an equal-weight reference, and a
  **live panel** tracks live-news rebalances forward against fresh market prices.
* **Module B - Strategic Portfolio Stress Testing.** High-impact events (impact > 7) trigger scenario shocks that
  revalue a wholesale banking book (loans, bonds, derivatives, equities) and report before/after P&L with a RAG
  status.

## 2. Architecture & Tech Stack

![Architecture](docs/architecture.png)

<details>
<summary>Text version of the data flow</summary>

```
 unstructured text              structured signals                 downstream
┌──────────────────┐          ┌────────────────────────┐         ┌──────────────────────┐
│ News headlines   │ ingest   │ Sentiment Score   -1..1│  store  │ Module A             │
│  · GDELT (live)  │────────▶ │ Event Class       enum │────────▶│  IndexRebalancer     │
│  · Google News   │          │ Impact Score       1-10│ sqlite  │  ±weight per signal  │
│  · bundled news  │          │ + ticker, confidence   │  +jsonl │  turnover cap/bounds │
│ Tweets (sample)  │          └────────────────────────┘  +API   └──────────┬───────────┘
└──────────────────┘                 │  NLP: local lexicon (default)       │ weight snapshots
                                     │       or OpenAI/Anthropic (hybrid)  ▼
                                     └────── FastAPI :8000 ◀── Streamlit dashboard :8501

   same signal stream ──▶ Module B: event class + impact > 7 ──▶ scenario shocks ──▶ revalue
                          the book (loans/bonds/derivatives/equities) ──▶ before/after + RAG status
```
</details>

**Data flow:** feeds → de-duplication (by id and story) → NLP analyzer → signal store → pub/sub → Module A / Module B
→ FastAPI → Streamlit dashboards.

**Tech stack**

| Layer | Choice | Why |
|---|---|---|
| Language | Python | Single stack for NLP, finance maths and the UI |
| NLP | Local lexicon/rule engine (default); optional OpenAI / Anthropic backend with automatic fallback | Deterministic, offline and free by default; LLM is opt-in and any failure falls back to local |
| API | FastAPI + Uvicorn | Typed, auto-generated docs at `/docs` |
| Storage | SQLite + JSONL | Zero-setup, inspectable, resumable |
| Dashboard | Streamlit + Plotly | Fast to build, interactive; Module A page auto-refreshes every 10 s |
| Data | pandas, NumPy, pydantic, requests, yfinance | Validation, back-test and live price tracking |
| Live feeds | GDELT -> Google News RSS -> Yahoo Finance RSS -> NewsAPI | First feed that returns headlines wins; retries, 429 handling and clear failure reasons |

## 3. Dataset Used

All data is **publicly available or synthetic**. No proprietary data and no S&P Global / Crisil client data is used.

| File | Source | Used for |
|---|---|---|
| `data/raw/stock_tweets.csv` | Kaggle - *Tweets about the Top Companies* [add URL] | Social-media signals (replay) |
| `data/raw/stock_yfinance_data.csv` | Same Kaggle bundle, daily prices (30 Sep 2021 - 29 Sep 2022, 25 tickers) | Back-test of Module A |
| `data/raw/all-data.csv` | Kaggle - *Sentiment Analysis for Financial News* [add URL] | News signals (replay) |
| `data/raw/FinancialPhraseBank/Sentences_AllAgree.txt` | Financial PhraseBank (Malo et al.) [add URL] | Validating the NLP model |
| `data/processed/loan_book.csv` | Derived from Kaggle - *Financial Transactions* [add URL] by `scripts/prepare_transactions.py` | Module B banking book (loans) |
| Generated in code | `module_b/portfolio.py` (seeded, synthetic) | Bonds, derivatives and equities in the book |
| Live headlines | GDELT and Google News RSS (public feeds) | Live mode only; not stored in the repo |

**Assumptions**

* Dataset licences differ; each is credited to its original author and used for demonstration only.
* PhraseBank sentences carry no date or ticker, so they are spread evenly across the tweet period and act as
  market-level events. They feed the engine and Module B, not Module A's per-stock weights.
* The loan book is *derived*: each client's card credit limit is allocated to (sector, US region) segments in
  proportion to their spending, rated by credit score and transaction error rate, and scaled to a $3.3bn wholesale
  book. It is a stand-in for a real loan book, not real exposures.
* Replay size is set by `DATASET_TWEET_LIMIT` (default 3000, balanced per ticker) and `DATASET_NEWS_LIMIT` (default 400).
* The raw 1.2 GB transactions file is **not** in the repo (GitHub's 100 MB limit). `loan_book.csv` is committed, so
  Module B runs without it.

## 4. Quickstart & Installation

Runtime: Python [3.x - run `python --version`] on [Windows 11 / PowerShell - confirm your OS]

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv && venv\Scripts\activate        # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python run_engine.py seed     # replay the bundled tweets + news offline (feeds Module A and B)
python run_api.py             # terminal 1: signal API   -> http://127.0.0.1:8000/docs
python run_dashboard.py       # terminal 2: dashboards   -> http://localhost:8501
```

The dashboard has three pages: **Module A** (home), **Stress Testing** and **Risk Engine** (Signal Lab). Use the
sidebar buttons *Replay Kaggle datasets* or *Ingest live news (GDELT/RSS)*. While `run_api.py` is running, ingest
through the dashboard or API rather than the CLI so the API's in-memory index weights stay in sync.

Other commands:

```bash
python run_engine.py analyze "JPMorgan misses estimates, credit downgrade sparks selloff"   # one headline
python run_engine.py diagnose          # test every live feed; explains DNS / proxy / TLS problems
python run_engine.py once --source live
python run_engine.py reset             # wipe signals, weights and stress runs
python scripts/evaluate.py             # NLP accuracy on PhraseBank + tweet signal vs real returns
python -m pytest tests/ -q             # 108 tests, fully offline
```

<details>
<summary>Live-feed troubleshooting</summary>

| Symptom | Fix |
|---|---|
| `DNS lookup failed` | No internet, or a VPN / proxy is needed |
| `TLS/certificate error` | Corporate proxy or antivirus intercepts HTTPS: `RISK_CA_BUNDLE=<corp-ca.pem>` (last resort `RISK_SSL_VERIFY=0`) |
| `proxy error` | Set `HTTPS_PROXY` / `HTTP_PROXY` |
| `HTTP 429` | Rate limited (GDELT allows 1 request / 5 s); wait and retry |
| `GDELT answered with text` | GDELT rejected a query. The engine sends 3 short batched queries and skips a rejected one; set `GDELT_QUERY` to use your own |

If GDELT fails, the engine falls back to Google News RSS automatically. `tests/mock_feeds.py` imitates all feeds so
the whole live path is tested offline.
</details>

<details>
<summary>Repository layout and configuration</summary>

```
risk_engine/   ingestion/ nlp/ models.py store.py pipeline.py api.py   <- the engine
module_a/      index.py rebalancer.py backtest.py live.py              <- rebalancer, back-test, live tracking
module_b/      portfolio.py scenarios.py valuation.py stress.py store.py api.py
dashboard/     app.py (Module A) pages/1_Stress_Testing.py pages/2_Risk_Engine.py common.py
scripts/       prepare_transactions.py evaluate.py
data/          raw/ (small public datasets)  processed/loan_book.csv
docs/          presentation.pdf  architecture.png
tests/         unit, API and offline live-feed tests
system.py      composition root: the only file that knows both modules
run_*.py       engine CLI, API server, dashboard launcher
```

Environment variables: `ENGINE_LLM`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GDELT_QUERY`, `GDELT_BATCH_PAUSE`,
`RSS_QUERY`, `GDELT_MAX_RECORDS`, `RISK_OUTPUT_DIR`, `RISK_DB_PATH`, `RISK_API_PORT`, `NEWSAPI_KEY`.
</details>

## 5. Key Results & Domain Impact

**What the prototype demonstrates**

* **Signal engine:** text in, structured signals out (sentiment, event class, impact, ticker), served by a REST API
  with a live Signal Lab page.
* **Module A back-test** (Sep 2021 - Sep 2022, real prices, no look-ahead):

  | Sentiment-driven index | Equal-weight | Excess return | Tracking error (ann.) |
  |---|---|---|---|
  | -25.62% | -28.56% | +2.94% | 2.03% |

  The 12-month window was a falling market, so read this as "lost less than the equal-weight index", not as
  alpha. The signal is not statistically proven (see Limitations).
* **Live index panel:** live-news rebalances are tracked forward against fresh Yahoo Finance prices, because the
  historical back-test cannot see headlines dated after its price file ends.
* **Module B:** a high-impact event (impact > 7) triggers scenario shocks and a before/after revaluation of the
  book with RAG status. P&L reconciles by risk factor and by asset class, and a cooldown prevents repeat triggers.
* **NLP validation** on Financial PhraseBank (4,846 sentences): accuracy **0.716**, macro-F1 **0.645**, against a
  majority-class baseline of 0.594. The lexicon was tuned using error analysis on the same data, so treat this as
  optimistic.

**Why it matters**

* **Speed:** index desks and risk teams react to a headline in seconds instead of after an analyst reads it.
* **One signal, many users:** the tactical (index) and strategic (bank book) views come from the same signals, so
  they are consistent and auditable.
* **Controlled automation:** weight bounds, a turnover cap and impact thresholds keep automated reactions from
  becoming extreme trades.
* **Explainable and cheap by default:** a deterministic, offline NLP model works with no API keys; an LLM is optional.

**Limitations & next steps**

* Tweets vs next-day return (64k tweets): mean daily cross-sectional IC about +0.02 (t = 0.7), **not
  statistically different from zero**. The sentiment signal does not yet predict returns.
* The NLP model is lexicon/rule based. A trained model such as FinBERT would score higher.
* Impact is a heuristic severity, not a trained model. Stress shocks are calibrated judgement, not regulatory
  scenarios, and valuation is first-order sensitivities (no full repricing, netting or collateral).
* The live index starts from the weights the engine held when live news began, and only the 15 index members move
  weights.
* Next steps: FinBERT or a fine-tuned model, a larger live history to test the signal properly, regulatory
  scenarios, and full repricing of derivatives.

## AI Usage

AI assistance (Claude) was used while building this project for [fill in honestly, e.g. code review, debugging,
the live-index panel, GDELT query fix, documentation]. The design decisions, data choices and results were reviewed
and are understood by the author. The NLP backend can optionally call OpenAI/Anthropic APIs at runtime, which is
off by default.

## License

MIT - see [`LICENSE`](LICENSE).