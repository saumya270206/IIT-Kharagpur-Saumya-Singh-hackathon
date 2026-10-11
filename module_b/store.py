"""Persistence for stress-test runs: same SQLite file as the engine, own table."""

from __future__ import annotations

import json
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stress_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT,
    trigger          TEXT,
    signal_id        TEXT,
    headline         TEXT,
    scenario_key     TEXT,
    scenario_name    TEXT,
    event_class      TEXT,
    impact           INTEGER,
    sentiment        REAL,
    ticker           TEXT,
    value_before     REAL,
    value_after      REAL,
    pnl              REAL,
    pnl_pct          REAL,
    capital_buffer   REAL,
    buffer_used_pct  REAL,
    status           TEXT,
    detail           TEXT
);
CREATE INDEX IF NOT EXISTS idx_stress_ts ON stress_runs(ts);
"""

_SCALAR = ("ts", "trigger", "signal_id", "headline", "scenario_key", "scenario_name",
           "event_class", "impact", "sentiment", "ticker", "value_before",
           "value_after", "pnl", "pnl_pct", "capital_buffer", "buffer_used_pct",
           "status")
_DETAIL = ("by_asset", "by_factor", "by_sector", "by_region", "top_positions",
           "position_pnl", "shock")


class StressStore:
    def __init__(self, signal_store) -> None:
        self._s = signal_store
        self.db_path = signal_store.db_path
        with self._s.connection() as con:
            con.executescript(_SCHEMA)

    def add(self, result) -> int:
        d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        detail = json.dumps({k: d.get(k) for k in _DETAIL})
        with self._s.connection() as con:
            cur = con.execute(
                f"INSERT INTO stress_runs ({','.join(_SCALAR)}, detail) "
                f"VALUES ({','.join('?' * (len(_SCALAR) + 1))})",
                [d.get(k) for k in _SCALAR] + [detail],
            )
            return int(cur.lastrowid)

    @staticmethod
    def _row(row, with_detail: bool) -> dict[str, Any]:
        out = {k: row[k] for k in ("id", *_SCALAR)}
        if with_detail:
            out.update(json.loads(row["detail"] or "{}"))
        return out

    def list(self, limit: int = 50, with_detail: bool = False,
             trigger: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM stress_runs", []
        if trigger:
            sql += " WHERE trigger = ?"
            args.append(trigger)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._s.connection() as con:
            rows = con.execute(sql, args).fetchall()
        return [self._row(r, with_detail) for r in rows]

    def get(self, run_id: int) -> Optional[dict]:
        with self._s.connection() as con:
            row = con.execute("SELECT * FROM stress_runs WHERE id = ?",
                              (run_id,)).fetchone()
        return self._row(row, True) if row else None

    def latest(self, with_detail: bool = True) -> Optional[dict]:
        rows = self.list(limit=1, with_detail=with_detail)
        return rows[0] if rows else None

    def count(self) -> int:
        with self._s.connection() as con:
            return int(con.execute("SELECT COUNT(*) FROM stress_runs").fetchone()[0])

    def clear(self) -> None:
        with self._s.connection() as con:
            con.execute("DELETE FROM stress_runs")
