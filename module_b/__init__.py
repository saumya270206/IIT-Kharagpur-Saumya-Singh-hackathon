"""Module B: strategic, event-driven portfolio stress testing.

Consumes the Event Classification and Impact Score from the NLP Risk Engine.
When a high-impact adverse event is detected it revalues a synthetic wholesale
banking book (loans, bonds, derivatives, equities) under a scenario-specific
shock set and records the before/after impact.
"""

from module_b.portfolio import Portfolio, Position, load_portfolio
from module_b.scenarios import TEMPLATES, Scenario, build_scenario
from module_b.store import StressStore
from module_b.stress import StressConfig, StressEngine, StressResult, StressTester

__all__ = [
    "Portfolio", "Position", "load_portfolio", "TEMPLATES", "Scenario",
    "build_scenario", "StressStore", "StressConfig", "StressEngine",
    "StressResult", "StressTester",
]
