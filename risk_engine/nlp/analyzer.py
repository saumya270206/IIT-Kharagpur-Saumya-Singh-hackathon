"""Hybrid NLP analyzer: optional LLM backend with automatic local fallback.

``RiskAnalyzer`` is what the pipeline calls. It turns a ``RawItem`` into a
``RiskSignal`` using one of:

* ``local``      - the offline lexicon/rule model in ``risk_engine.nlp.local``
* ``openai``     - OpenAI chat completions   (needs OPENAI_API_KEY)
* ``anthropic``  - Anthropic messages API    (needs ANTHROPIC_API_KEY)

Backend selection (``backend`` argument, default = env ``ENGINE_LLM``):

    auto       OpenAI if its key is set, else Anthropic if its key is set, else local
    off/local  always local (deterministic, no network)
    openai / anthropic
               that LLM; if its key is missing the analyzer degrades to local

Any LLM failure (network error, bad JSON, out-of-range values) falls back to
the local model for that item, so the feed never stops. After a failure the LLM
is skipped for a short cool-down so a dead endpoint cannot slow every item.

``RiskAnalyzer.backend`` is the *configured* backend ("local", "openai" or
"anthropic"); each signal's own ``backend`` field records what actually scored it.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

import requests

from risk_engine import config
from risk_engine.models import EventClass, RawItem, RiskSignal, utcnow
from risk_engine.nlp.local import LocalAnalyzer, resolve_entity

log = logging.getLogger("risk_engine.analyzer")

BACKENDS = ("auto", "off", "local", "openai", "anthropic")

LLM_TIMEOUT = 20.0          # seconds per request
FAILURE_COOLDOWN = 60.0     # seconds to skip the LLM after a failure

_SYSTEM_PROMPT = (
    "You are a financial risk analyst. Read the text (a news item or social media "
    "post) and return ONLY a JSON object, no prose and no markdown, with keys:\n"
    '  "sentiment": number from -1.0 (very negative for the named company/market) '
    "to 1.0 (very positive), 0 if neutral,\n"
    '  "event_class": one of {classes},\n'
    '  "impact": integer 1-10, the likely market-impact severity '
    "(1 = negligible, 10 = systemic shock),\n"
    '  "confidence": number 0.0-1.0.'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _resolve_backend(requested: Optional[str]) -> str:
    """Map the requested mode to the backend that will actually be used."""
    mode = (requested or config.ENGINE_LLM or "auto").lower()
    if mode not in BACKENDS:
        log.warning("unknown ENGINE_LLM %r; using local", mode)
        return "local"
    if mode in ("off", "local"):
        return "local"
    if mode == "auto":
        if config.OPENAI_API_KEY:
            return "openai"
        if config.ANTHROPIC_API_KEY:
            return "anthropic"
        return "local"
    # explicit LLM choice, but only if its key exists
    key = config.OPENAI_API_KEY if mode == "openai" else config.ANTHROPIC_API_KEY
    if not key:
        log.warning("ENGINE_LLM=%s but no API key set; using local", mode)
        return "local"
    return mode


def _parse_llm_json(raw: str) -> dict:
    """Extract and validate the model's JSON answer. Raises ValueError if unusable."""
    match = _JSON_RE.search(raw or "")
    if not match:
        raise ValueError("no JSON object in LLM reply")
    data = json.loads(match.group(0))

    sentiment = float(data["sentiment"])
    impact = int(round(float(data["impact"])))
    confidence = float(data.get("confidence", 0.6))
    event = EventClass(str(data["event_class"]).strip())  # ValueError if not a known class

    if not -1.0 <= sentiment <= 1.0:
        raise ValueError(f"sentiment out of range: {sentiment}")
    if not 1 <= impact <= 10:
        raise ValueError(f"impact out of range: {impact}")
    return {
        "sentiment": round(sentiment, 3),
        "event_class": event,
        "impact": impact,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
    }


class RiskAnalyzer:
    """Pipeline-facing analyzer: LLM when available, local model otherwise."""

    def __init__(self, backend: Optional[str] = None) -> None:
        self.backend = _resolve_backend(backend)
        self._local = LocalAnalyzer()
        self._llm_down_until = 0.0
        log.info("NLP backend: %s", self.backend)

    # ------------------------------------------------------------------ API
    def analyze(self, item: RawItem) -> RiskSignal:
        if self.backend != "local" and time.monotonic() >= self._llm_down_until:
            try:
                return self._analyze_llm(item)
            except Exception as exc:  # noqa: BLE001 - never let the LLM stop the feed
                self._llm_down_until = time.monotonic() + FAILURE_COOLDOWN
                log.warning("%s backend failed (%s); using local model for %ds",
                            self.backend, exc, int(FAILURE_COOLDOWN))
        return self._local.analyze(item)

    # ------------------------------------------------------------ LLM paths
    def _analyze_llm(self, item: RawItem) -> RiskSignal:
        raw = self._call_openai(item.text) if self.backend == "openai" \
            else self._call_anthropic(item.text)
        parsed = _parse_llm_json(raw)
        ticker, company = resolve_entity(item.text, item.ticker, item.company)
        return RiskSignal(
            signal_id=uuid.uuid4().hex,
            item_id=item.id,
            source=item.source,
            text=item.text,
            url=item.url,
            ticker=ticker,
            company=company,
            sentiment=parsed["sentiment"],
            event_class=parsed["event_class"],
            impact=parsed["impact"],
            confidence=parsed["confidence"],
            backend=self.backend,
            event_at=item.published_at,
            created_at=utcnow(),
        )

    @staticmethod
    def _system_prompt() -> str:
        classes = ", ".join(f'"{e.value}"' for e in EventClass)
        return _SYSTEM_PROMPT.replace("{classes}", classes)

    def _call_openai(self, text: str) -> str:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={
                "model": config.OPENAI_MODEL,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": text[:4000]},
                ],
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_anthropic(self, text: str) -> str:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": config.ANTHROPIC_MODEL,
                "max_tokens": 200,
                "temperature": 0,
                "system": self._system_prompt(),
                "messages": [{"role": "user", "content": text[:4000]}],
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json()["content"])