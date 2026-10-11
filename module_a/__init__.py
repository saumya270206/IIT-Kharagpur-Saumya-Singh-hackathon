"""Module A: tactical, high-frequency stock index rebalancing.

Consumes RiskSignals emitted by the NLP Risk Engine and adjusts the weights
of a mock index built from S&P 100 constituents.
"""

from module_a.index import INDEX_TICKERS, equal_weights
from module_a.rebalancer import IndexRebalancer, RebalanceConfig

__all__ = ["INDEX_TICKERS", "equal_weights", "IndexRebalancer", "RebalanceConfig"]
