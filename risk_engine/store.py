"""Persistence layer shared by the engine and every downstream module.

* SQLite (``data/output/risk.db``)  - ``signals`` and ``weights`` tables
* JSON Lines (``data/output/signals.jsonl``) - every signal, append-only

The schema is identical to the one already shipped in ``risk.db`` so existing
databases keep working. Connections are opened per operation (WAL mode), which
makes the store safe to share between FastAPI worker threads, the Streamlit
dashboard and the CLI at the same time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Sequence

from risk_engine import config
from risk_engine.models import RiskSignal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id   TEXT PRIMARY KEY,
    item_id     TEXT,
    source      TEXT,
    text        TEXT,
    url         TEXT,
    ticker      TEXT,
    company     TEXT,
    sentiment   REAL,
    event_class TEXT,
    impact      INTEGER,
    confidence  REAL,
    backend     TEXT,
    event_at    TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_event_at ON signals(event_at);
CREATE INDEX IF NOT EXISTS idx_signals_ticker   ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_item     ON signals(item_id);

CREATE TABLE IF NOT EXISTS weights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    ticker      TEXT,
    weight      REAL,
    signal_id   TEXT,
    sentiment   REAL,
    event_class TEXT,
    impact      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_weights_ts ON weights(ts);
"""

_SIGNAL_COLS = (
    "signal_id", "item_id", "source", "text", "url", "ticker", "company",
    "sentiment", "event_class", "impact", "confidence", "backend",
    "event_at", "created_at",
)


def _iso(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


class SignalStore:
    """Thread-safe SQLite store (+ JSONL mirror) for signals and weights."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        jsonl_path: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else Path(config.DB_PATH)
        default_jsonl = (
            config.SIGNALS_JSONL
            if db_path is None
            else self.db_path.with_name("signals.jsonl")
        )
        self.jsonl_path = Path(jsonl_path) if jsonl_path else Path(default_jsonl)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connection() as con:
            con.executescript(_SCHEMA)

    # ------------------------------------------------------------ connection
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection; commits on success, rolls back on error.
        Downstream modules reuse this to keep their own tables in the same DB."""
        with self._lock:
            con = sqlite3.connect(self.db_path, timeout=30)
            con.row_factory = sqlite3.Row
            try:
                con.execute("PRAGMA journal_mode=WAL")
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()

    # --------------------------------------------------------------- signals
    def add_signal(self, signal: RiskSignal) -> bool:
        """Persist a signal (SQLite + JSONL). Returns False if already stored."""
        row = signal.to_json()
        with self.connection() as con:
            cur = con.execute(
                f"INSERT OR IGNORE INTO signals ({','.join(_SIGNAL_COLS)}) "
                f"VALUES ({','.join('?' * len(_SIGNAL_COLS))})",
                [row[c] if c != "event_class" else signal.event_class.value
                 for c in _SIGNAL_COLS],
            )
            inserted = cur.rowcount > 0
        if inserted:
            with self._lock, open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return inserted

    def has_item(self, item_id: str) -> bool:
        with self.connection() as con:
            r = con.execute("SELECT 1 FROM signals WHERE item_id = ? LIMIT 1",
                            (item_id,)).fetchone()
        return r is not None

    def count_signals(self) -> int:
        with self.connection() as con:
            return int(con.execute("SELECT COUNT(*) FROM signals").fetchone()[0])

    @staticmethod
    def _to_signal(row: sqlite3.Row) -> RiskSignal:
        return RiskSignal(**{k: row[k] for k in _SIGNAL_COLS})

    def get_signals(
        self,
        limit: int = 50,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[datetime | str] = None,
        event_class: Optional[str] = None,
    ) -> list[RiskSignal]:
        """Newest first. ``since`` filters on ``created_at`` (when the engine
        emitted the signal) so consumers can poll for what they have not seen."""
        where, args = [], []
        if ticker:
            where.append("ticker = ?")
            args.append(ticker.upper())
        if source:
            where.append("source = ?")
            args.append(source)
        if event_class:
            where.append("event_class = ?")
            args.append(event_class)
        if since:
            where.append("created_at >= ?")
            args.append(_iso(since))
        sql = "SELECT * FROM signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY rowid DESC LIMIT ?"
        args.append(int(limit))
        with self.connection() as con:
            rows = con.execute(sql, args).fetchall()
        return [self._to_signal(r) for r in rows]

    # --------------------------------------------------------------- weights
    def add_weight_snapshot(self, rows: Sequence[dict]) -> None:
        with self.connection() as con:
            con.executemany(
                "INSERT INTO weights (ts, ticker, weight, signal_id, sentiment, "
                "event_class, impact) VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        _iso(r["ts"]), r["ticker"], float(r["weight"]),
                        r.get("signal_id"), r.get("sentiment"),
                        r.get("event_class"), r.get("impact"),
                    )
                    for r in rows
                ],
            )

    def snapshot_times(self) -> list[str]:
        """Distinct snapshot timestamps in insertion (= chronological) order."""
        with self.connection() as con:
            rows = con.execute(
                "SELECT ts FROM weights GROUP BY ts ORDER BY MIN(id)"
            ).fetchall()
        return [r["ts"] for r in rows]

    def latest_weights(self) -> dict[str, float]:
        with self.connection() as con:
            last = con.execute(
                "SELECT ts FROM weights ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if last is None:
                return {}
            rows = con.execute(
                "SELECT ticker, weight FROM weights WHERE ts = ? ORDER BY id",
                (last["ts"],),
            ).fetchall()
        return {r["ticker"]: float(r["weight"]) for r in rows}

    def get_weight_history(self, since: Optional[datetime | str] = None) -> list[dict]:
        sql = ("SELECT ts, ticker, weight, signal_id, sentiment, event_class, impact "
               "FROM weights")
        args: list = []
        if since:
            sql += " WHERE ts >= ?"
            args.append(_iso(since))
        sql += " ORDER BY id"
        with self.connection() as con:
            rows = con.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- admin
    def reset(self) -> None:
        """Delete all signals/weights (and the JSONL mirror). Dev/test helper."""
        with self.connection() as con:
            con.execute("DELETE FROM signals")
            con.execute("DELETE FROM weights")
        with self._lock:
            self.jsonl_path.write_text("", encoding="utf-8")
