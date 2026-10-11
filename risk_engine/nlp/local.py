"""Local, offline NLP model: sentiment scoring, event classification, impact.

Deterministic lexicon/rule based model (no downloads, no network). It is the
default backend and the fallback whenever the LLM backend is unavailable.
"""

from __future__ import annotations

import math
import re
import uuid
from typing import Optional

from risk_engine.companies import match_company
from risk_engine.models import EventClass, RiskSignal, RawItem, utcnow
from risk_engine.nlp.lexicon import (
    DIMINISHERS,
    EVENT_KEYWORDS,
    IMPACT_BASE,
    INTENSIFIERS,
    INTENSITY_STEMS,
    NEGATORS,
    PHRASE_LEXICON,
    SENTIMENT_LEXICON,
)

_TOKEN_RE = re.compile(r"[a-z0-9$']+[.!]?|[%]", re.IGNORECASE)
_INFLECT = r"(?:s|es|ed|d|ing|er|ers)?"


def _stem_pattern(stem: str) -> str:
    """Regex for a keyword stem.

    Long stems keep prefix semantics ("downgrad" -> downgraded). Short single
    words are matched as whole words plus common inflections only, otherwise
    "war" fires on "warm", "ban" on "bank", "fine" on "finance", "coup" on
    "coupon" and "fed" on "federal".
    """
    if len(stem) <= 4 and stem.isalnum():
        # optional doubled final consonant: ban -> banned / banning
        return r"\b" + re.escape(stem) + re.escape(stem[-1]) + r"?" + _INFLECT + r"\b"
    return r"\b" + re.escape(stem) + r"\w*"


_EVENT_PATTERNS: dict[EventClass, list[tuple[re.Pattern[str], float]]] = {
    cls: [(re.compile(_stem_pattern(stem), re.IGNORECASE), w) for stem, w in pairs]
    for cls, pairs in EVENT_KEYWORDS.items()
}
_INTENSITY_PATTERNS = [
    re.compile(_stem_pattern(stem) if len(stem) <= 4 and stem.isalnum()
               else re.escape(stem), re.IGNORECASE)
    for stem in INTENSITY_STEMS
]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def score_sentiment(text: str) -> float:
    """Return sentiment in [-1.0, 1.0]."""
    toks = _tokens(text)
    lower = [t.lower().rstrip(".!") for t in toks]
    raw = 0.0
    consumed: set[int] = set()

    # multi-word phrases first (consume both tokens)
    for i in range(len(lower) - 1):
        if i in consumed:
            continue
        bigram = f"{lower[i]} {lower[i+1]}"
        if bigram in PHRASE_LEXICON:
            raw += PHRASE_LEXICON[bigram]
            consumed.update({i, i + 1})

    # negation scopes: a negator flips polarity of tokens within +/-3 words
    # of it, bounded to its clause ("profits did not rise" -> negative).
    clause_ids: list[int] = []
    cid = 0
    for t in toks:
        clause_ids.append(cid)
        if t.endswith((".", "!")):
            cid += 1
    negated = [False] * len(lower)
    for i, tok in enumerate(lower):
        if tok in NEGATORS and tok not in SENTIMENT_LEXICON:
            for j in range(len(lower)):
                if clause_ids[j] == clause_ids[i] and abs(j - i) <= 3:
                    negated[j] = True

    for i, tok in enumerate(lower):
        if i in consumed:
            continue
        val = SENTIMENT_LEXICON.get(tok)
        if val is None:
            continue
        if negated[i]:
            val *= -0.75  # negation flips (but less strongly)
        for j in range(max(0, i - 2), i):
            if lower[j] in INTENSIFIERS:
                val *= INTENSIFIERS[lower[j]]
            elif lower[j] in DIMINISHERS:
                val *= DIMINISHERS[lower[j]]
        if toks[i] != tok and toks[i].isupper() and len(toks[i]) > 1:
            val *= 1.15  # ALL CAPS emphasis
        raw += val

    exclaims = min(text.count("!"), 3) * 0.15
    raw += exclaims if raw != 0 else 0.0

    if raw == 0.0:
        return 0.0
    norm = raw / math.sqrt(raw * raw + 15.0) * 1.15
    return max(-1.0, min(1.0, norm))


def classify_event(text: str) -> tuple[EventClass, float]:
    """Return (event class, confidence in 0..1)."""
    scores: dict[EventClass, float] = {}
    for cls, patterns in _EVENT_PATTERNS.items():
        total = 0.0
        for pat, w in patterns:
            matches = pat.findall(text)
            if matches:
                total += w * (1.0 + 0.35 * (len(matches) - 1))
        if total:
            scores[cls] = total
    if not scores:
        return EventClass.OTHER, 0.30
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cls, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = 0.5 + 0.5 * (top - second) / top if top else 0.5
    return top_cls, round(max(0.3, min(1.0, confidence)), 3)


def score_impact(text: str, sentiment: float, event: EventClass,
                 event_confidence: float) -> int:
    """Return predicted market impact severity, 1..10."""
    score = IMPACT_BASE[event]
    lowered = text.lower()

    hits = sum(1 for pat in _INTENSITY_PATTERNS if pat.search(lowered))
    score += min(2.4, 0.55 * hits)
    score += min(1.5, abs(sentiment) * 1.5)
    score += min(1.0, max(0.0, (event_confidence - 0.5)) * 2.0)

    if text.isupper() or "!" in text:
        score += 0.4
    if re.search(r"\b(urgently|breaking|exclusive|halted|immediately)\b", lowered):
        score += 0.6

    return int(max(1, min(10, round(score))))


def resolve_entity(text: str, hint_ticker: Optional[str],
                   hint_company: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Ticker/company resolution: explicit hint > cashtag > name aliases."""
    if hint_ticker:
        return hint_ticker.upper(), hint_company
    return match_company(text)


class LocalAnalyzer:
    """Produces a full RiskSignal from a RawItem using static rules only."""

    name = "local"

    def analyze(self, item: RawItem) -> RiskSignal:
        sentiment = score_sentiment(item.text)
        event, confidence = classify_event(item.text)
        impact = score_impact(item.text, sentiment, event, confidence)
        ticker, company = resolve_entity(item.text, item.ticker, item.company)
        return RiskSignal(
            signal_id=uuid.uuid4().hex,
            item_id=item.id,
            source=item.source,
            text=item.text,
            url=item.url,
            ticker=ticker,
            company=company,
            sentiment=round(sentiment, 3),
            event_class=event,
            impact=impact,
            confidence=confidence,
            backend=self.name,
            event_at=item.published_at,
            created_at=utcnow(),
        )
