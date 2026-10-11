"""CLI for the risk engine.

    python run_engine.py seed                 # replay the Kaggle datasets (offline)
    python run_engine.py once --source live   # one live cycle (GDELT -> RSS fallback)
    python run_engine.py loop --source both --interval 900
    python run_engine.py analyze "JPMorgan misses estimates, credit downgrade"
    python run_engine.py reset                # wipe signals, weights and stress runs
    python run_engine.py diagnose             # test every live feed and explain failures

Note: when run_api.py is already running, trigger ingestion through the
dashboard buttons or POST /pipeline/run so the API's in-memory index state
stays in sync.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from risk_engine.models import RawItem, utcnow
from system import build_system


def _sources(name: str) -> list[str]:
    return {"seed": ["datasets"], "datasets": ["datasets"], "live": ["live"],
            "gdelt": ["gdelt"], "rss": ["rss"], "yahoo": ["yahoo"],
            "newsapi": ["newsapi"], "both": ["datasets", "live"]}[name]


def main() -> None:
    ap = argparse.ArgumentParser(description="AI/NLP Risk Engine CLI")
    ap.add_argument("command", choices=["seed", "once", "loop", "analyze", "reset", "diagnose"])
    ap.add_argument("text", nargs="?", help="text for the analyze command")
    ap.add_argument("--source", default="datasets",
                    choices=["datasets", "live", "gdelt", "rss", "yahoo", "newsapi", "both"])
    ap.add_argument("--interval", type=int, default=900, help="loop seconds")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if args.command == "diagnose":
        from risk_engine.ingestion.diagnose import run_diagnostics
        raise SystemExit(run_diagnostics())

    sys_ = build_system()

    if args.command == "reset":
        sys_.store.reset()
        sys_.stress.results.clear()
        print("cleared signals, weights and stress runs")
        return

    if args.command == "analyze":
        if not args.text:
            ap.error("analyze needs a text argument")
        sig = sys_.pipeline.process(RawItem(
            id=f"manual-{utcnow().timestamp()}", source="manual", text=args.text))
        print(json.dumps(sig.to_json() if sig else {"error": "duplicate"}, indent=2))
        return

    sources = _sources("datasets" if args.command == "seed" else args.source)

    def cycle() -> None:
        s = sys_.pipeline.run_cycle(sources)
        print(json.dumps(s))
        print(f"  stress tests recorded: {sys_.stress.results.count()}  |  "
              f"rebalance events: {len(sys_.store.snapshot_times())}  |  "
              f"signals: {sys_.store.count_signals()}")

    if args.command in ("seed", "once"):
        cycle()
        return
    while True:  # loop
        cycle()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()