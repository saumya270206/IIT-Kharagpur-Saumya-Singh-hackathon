"""Core data structures shared by the engine and downstream modules."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventClass(str, Enum):
    """Categorical label for the type of event discussed in a text item."""

    GEOPOLITICAL = "Geopolitical"
    MACROECONOMIC = "Macroeconomic"
    CREDIT_EVENT = "Credit Event"
    MERGER_ACQUISITION = "Merger/Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    EARNINGS = "Earnings"
    REGULATORY = "Regulatory"
    OPERATIONAL = "Operational"
    OTHER = "Other"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RawItem(BaseModel):
    """A single unstructured text item from one of the ingestion sources."""

    id: str
    source: str  # "news" | "twitter" | "manual"
    text: str
    url: Optional[str] = None
    published_at: datetime = Field(default_factory=utcnow)
    ticker: Optional[str] = None
    company: Optional[str] = None


class RiskSignal(BaseModel):
    """Structured, machine-readable risk signal emitted by the engine."""

    signal_id: str
    item_id: str
    source: str
    text: str
    url: Optional[str] = None
    ticker: Optional[str] = None
    company: Optional[str] = None
    sentiment: float = Field(ge=-1.0, le=1.0, description="Positive/negative/neutral, -1.0 to 1.0")
    event_class: EventClass
    impact: int = Field(ge=1, le=10, description="Predicted market impact severity, 1 to 10")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    backend: str = Field(default="local", description="NLP backend that produced the signal")
    event_at: datetime = Field(default_factory=utcnow, description="Publication time of the source item")
    created_at: datetime = Field(default_factory=utcnow, description="Time the signal was generated")

    def to_json(self) -> dict:
        return self.model_dump(mode="json")
