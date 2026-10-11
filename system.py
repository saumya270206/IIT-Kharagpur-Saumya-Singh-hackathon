"""Composition root: wires the engine to Module A and Module B.

The engine itself never imports its consumers; this file is the only place
that knows about both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

from module_a import IndexRebalancer
from module_b import StressTester
from risk_engine.pipeline import Pipeline
from risk_engine.store import SignalStore


@dataclass
class System:
    store: SignalStore
    pipeline: Pipeline
    rebalancer: IndexRebalancer
    stress: StressTester


def build_system(db_path: str | None = None) -> System:
    store = SignalStore(db_path)
    pipeline = Pipeline(store)
    rebalancer = IndexRebalancer(store)
    stress = StressTester(store)
    pipeline.subscribe(rebalancer.apply)   # Module A: sentiment -> weights
    pipeline.subscribe(stress.apply)       # Module B: event class + impact -> stress test
    return System(store, pipeline, rebalancer, stress)
