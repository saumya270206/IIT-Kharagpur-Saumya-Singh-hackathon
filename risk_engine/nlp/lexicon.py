"""Finance-tuned lexicons and rules driving the local NLP model.

Everything here is static reference data so the local backend is fully
deterministic, offline and needs no model downloads.
"""

from __future__ import annotations

from risk_engine.models import EventClass

# ------------------------------------------------------------------------- sentiment
# Single-token sentiment valences (roughly VADER-style: magnitude ~ strength).
SENTIMENT_LEXICON: dict[str, float] = {
    # positives
    "surge": 2.5, "surges": 2.5, "surged": 2.5, "surging": 2.5,
    "rally": 2.4, "rallies": 2.4, "rallied": 2.4, "rallying": 2.4,
    "soar": 2.6, "soars": 2.6, "soared": 2.6, "soaring": 2.6,
    "jump": 2.0, "jumps": 2.0, "jumped": 2.0, "jumping": 2.0,
    "climb": 1.6, "climbs": 1.6, "climbed": 1.6,
    "record": 1.5, "beat": 2.0, "beats": 2.0, "beating": 2.0, "beaten": 1.6,
    "strong": 1.8, "strongest": 2.0, "strength": 1.5, "robust": 1.9,
    "solid": 1.6, "resilient": 1.5, "resilience": 1.3,
    "growth": 1.7, "grew": 1.5, "expand": 1.3, "expansion": 1.5,
    "profit": 1.5, "profits": 1.5, "profitable": 1.4, "profitability": 1.2,
    "gain": 1.4, "gains": 1.4, "gained": 1.4,
    "rise": 1.2, "rises": 1.2, "rose": 1.6, "rising": 1.2,
    "bullish": 2.4, "bull": 1.6, "upgrade": 1.8, "upgraded": 1.8, "upgrades": 1.8,
    "outperform": 1.8, "outperforms": 1.8, "overweight": 1.4,
    "buy": 1.2, "accumulate": 1.2, "innovative": 1.6, "innovation": 1.4,
    "unveil": 1.3, "unveils": 1.3, "unveiled": 1.3, "launch": 0.8, "launches": 0.8,
    "award": 1.2, "awards": 1.2, "wins": 1.6, "won": 1.4, "victory": 1.6,
    "approval": 1.6, "approved": 1.6, "approves": 1.6, "clearance": 1.3,
    "breakthrough": 2.0, "partnership": 1.3, "partners": 1.2,
    "buyback": 1.5, "dividend": 1.0, "dividends": 1.0, "special dividend": 1.8,
    "exceeded": 1.8, "exceeds": 1.6, "optimistic": 1.8, "optimism": 1.5,
    "confident": 1.3, "confidence": 1.0, "rebound": 1.6, "rebounds": 1.6,
    "recovery": 1.5, "recovering": 1.3, "improve": 1.2, "improves": 1.2,
    "improved": 1.2, "improvement": 1.2, "high": 0.6, "highs": 1.0,
    "winning": 1.6, "winner": 1.4, "upside": 1.4, "opportunity": 1.0,
    "moon": 1.8, "incredible": 2.0, "amazing": 1.8,
    "empire": 0.6, "thrive": 1.6, "thriving": 1.6, "beat": 2.0,
    "positive": 1.4, "positives": 1.4, "higher": 1.4, "upgrade": 1.8,
    # negatives
    "plunge": -2.8, "plunges": -2.8, "plunged": -2.8, "plunging": -2.8,
    "tumble": -2.4, "tumbles": -2.4, "tumbled": -2.4,
    "slump": -2.4, "slumps": -2.4, "slumped": -2.4,
    "slide": -1.8, "slides": -1.8, "slid": -1.8, "sliding": -1.8,
    "sink": -1.8, "sinks": -1.8, "sank": -1.8,
    "fall": -1.4, "falls": -1.4, "fell": -1.6, "falling": -1.4,
    "drop": -1.6, "drops": -1.6, "dropped": -1.6, "dropping": -1.6,
    "downturn": -2.0, "selloff": -2.4, "sell-off": -2.4,
    "miss": -2.0, "misses": -2.0, "missing": -1.8, "missed": -2.0, "shortfall": -2.2,
    "weak": -2.0, "weakness": -2.0, "weakest": -2.2, "soft": -0.8,
    "decline": -1.6, "declines": -1.6, "declined": -1.6, "declining": -1.6,
    "loss": -1.8, "losses": -1.8, "lost": -1.4,
    "lawsuit": -2.2, "litigation": -2.0, "sue": -2.2, "sues": -2.2, "sued": -2.2,
    "probe": -2.2, "probes": -2.2, "investigation": -2.0, "investigate": -1.8,
    "fraud": -2.8, "fraudulent": -2.8, "scandal": -2.6, "corruption": -2.4,
    "bankruptcy": -3.0, "bankrupt": -3.0, "bankruptcies": -3.0,
    "default": -2.8, "defaults": -2.8, "defaulted": -3.0, "insolvent": -3.0,
    "insolvency": -3.0, "downgrade": -2.2, "downgraded": -2.2, "downgrades": -2.2,
    "bearish": -2.4, "bear": -1.6, "underperform": -1.8, "underperforms": -1.8,
    "cut": -0.6, "cuts": -0.8, "cutting": -0.6, "slash": -1.6, "slashes": -1.6,
    "recall": -2.4, "recalls": -2.4, "recalled": -2.4,
    "outage": -2.2, "outages": -2.2, "breach": -2.4, "breached": -2.2,
    "hack": -2.4, "hacked": -2.4, "hacker": -2.4, "cyberattack": -2.6,
    "layoffs": -2.2, "layoff": -2.2, "redundancies": -2.0,
    "fine": -1.4, "fined": -1.8, "penalty": -1.8, "penalties": -1.6,
    "ban": -2.0, "bans": -2.0, "banned": -2.0, "prohibit": -1.8,
    "warning": -1.6, "warns": -1.6, "warn": -1.6, "warned": -1.6,
    "risk": -1.0, "risks": -1.2, "risky": -1.4, "peril": -1.6,
    "fear": -1.8, "fears": -1.8, "feared": -1.6, "uncertainty": -1.4,
    "volatile": -1.4, "volatility": -1.4, "uncertain": -1.3,
    "recession": -2.4, "recessionary": -2.4, "crisis": -2.6, "crises": -2.6,
    "collapse": -3.0, "collapses": -3.0, "collapsing": -3.0, "crash": -2.8, "crashes": -2.8,
    "controversy": -1.8, "controversial": -1.6,
    "charge": -1.0, "charges": -1.4, "charged": -1.8, "guilty": -2.4,
    "halt": -2.0, "halts": -2.0, "halted": -2.2, "suspend": -2.0, "suspends": -2.0,
    "suspended": -2.2, "shutdown": -2.4, "shut down": -2.2,
    "delay": -1.2, "delays": -1.4, "delayed": -1.4, "postpone": -1.4,
    "disappoint": -2.0, "disappointing": -2.0, "disappointed": -2.0,
    "struggle": -1.8, "struggles": -1.8, "struggling": -1.8,
    "pressure": -1.4, "pressured": -1.4, "crushing": -1.6, "crush": -1.4,
    "rage": -1.6, "furious": -1.8, "outrage": -1.8, "embarrassing": -1.8,
    "fail": -2.0, "fails": -2.0, "failed": -2.0, "failure": -2.2,
    "negative": -1.4, "lower": -1.2, "lowered": -1.2, "hurt": -1.6,
    "disaster": -2.6, "nightmare": -2.4, "garbage": -2.2, "disastrous": -2.8,
    "exit": -0.8, "withdraw": -1.2, "pullback": -1.4, "curbs": -1.2, "curb": -1.0,
}

# multi-word phrases (matched before single tokens; consume both tokens)
PHRASE_LEXICON: dict[str, float] = {
    "record high": 1.8, "record low": -2.0, "all-time high": 2.2, "all-time low": -2.2,
    "record revenue": 2.4, "record profit": 2.4, "record profits": 2.4,
    "profit warning": -2.4, "guidance cut": -2.4, "cuts guidance": -2.4,
    "raises guidance": 2.2, "raises outlook": 2.2, "lifts outlook": 2.2,
    "beats estimates": 2.2, "missed estimates": -2.2, "misses estimates": -2.2,
    "beat estimates": 2.2, "in line": 0.0, "job cuts": -2.2, "mass layoffs": -2.6,
    "filed for bankruptcy": -3.2, "missed payment": -2.6, "debt default": -3.0,
    "data breach": -2.6, "cyber attack": -2.6, "trade war": -2.4,
    "interest rate": 0.0, "rate cut": 0.8, "rate hike": -1.0, "rate hikes": -1.0,
    "share buyback": 1.8, "stock buyback": 1.8, "special dividend": 2.0,
    "under pressure": -1.8, "in trouble": -2.2, "at risk": -1.4,
    "record demand": 2.4, "strong demand": 2.0, "weak demand": -2.0,
    "price target": 0.0, "wall street": 0.0, "game changer": 2.2,
    "disaster": -2.6, "under water": -2.0,
}

# Domain polarity: a rising *risk* quantity is bad news, a falling one is good
# news - the opposite of what the generic verb alone ("surges", "falls") says.
_RISING = (
    "surge", "surges", "surged", "surging", "spike", "spikes", "spiked", "spiking",
    "jump", "jumps", "jumped", "soar", "soars", "soared", "soaring", "rise", "rises",
    "rose", "rising", "climb", "climbs", "climbed", "accelerate", "accelerates",
    "accelerated", "widen", "widens", "widened", "mount", "mounts", "mounted",
)
_FALLING = (
    "cool", "cools", "cooled", "cooling", "ease", "eases", "eased", "easing",
    "fall", "falls", "fell", "falling", "slow", "slows", "slowed", "decline",
    "declines", "declined", "drop", "drops", "dropped", "narrow", "narrows",
    "narrowed", "shrink", "shrinks", "shrank",
)
_BAD_WHEN_RISING = {
    "inflation": -2.2, "yields": -1.8, "costs": -2.0, "losses": -2.6, "defaults": -2.8,
    "volatility": -2.0, "unemployment": -2.2, "debt": -2.0, "spreads": -2.0,
    "delinquencies": -2.4, "layoffs": -2.4, "deficit": -2.0, "bankruptcies": -2.6,
}
_GOOD_WHEN_FALLING = {
    "inflation": 1.6, "yields": 0.8, "costs": 1.6, "losses": 2.0, "defaults": 2.2,
    "volatility": 1.4, "unemployment": 1.8, "debt": 1.4, "spreads": 1.4,
    "delinquencies": 1.8, "layoffs": 1.6, "deficit": 1.6,
}
for _noun, _val in _BAD_WHEN_RISING.items():
    for _verb in _RISING:
        PHRASE_LEXICON.setdefault(f"{_noun} {_verb}", _val)
for _noun, _val in _GOOD_WHEN_FALLING.items():
    for _verb in _FALLING:
        PHRASE_LEXICON.setdefault(f"{_noun} {_verb}", _val)

NEGATORS = {
    "not", "no", "never", "cannot", "cant", "don't", "dont", "doesn't", "doesnt",
    "won't", "wont", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "without", "failed", "fail", "fails", "failing", "barely", "hardly", "little",
}
INTENSIFIERS = {
    "very": 1.3, "extremely": 1.5, "sharply": 1.4, "strongly": 1.3, "significant": 1.3,
    "significantly": 1.4, "massive": 1.5, "hugely": 1.5, "record": 1.3, "staggering": 1.5,
    "dramatically": 1.4, "historic": 1.4, "soaring": 1.4, "plummeting": 1.4, "deep": 1.2,
    "major": 1.2, "stunning": 1.4, "incredible": 1.3, "blowout": 1.5,
}
DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "marginally": 0.6, "modest": 0.7, "modestly": 0.7,
    "little": 0.6, "mildly": 0.7, "tiny": 0.6, "narrowly": 0.7,
}

# ------------------------------------------------------------------------- events
# stem -> weight; matched as \bstem\w* (case-insensitive)
EVENT_KEYWORDS: dict[EventClass, list[tuple[str, float]]] = {
    EventClass.GEOPOLITICAL: [
        ("geopolitic", 2.5), ("war", 2.0), ("sanction", 2.2), ("invasion", 3.0),
        ("conflict", 1.8), ("military", 1.6), ("missile", 2.2), ("nato", 2.2),
        ("tariff", 1.8), ("embargo", 2.4), ("coup", 2.6), ("escalat", 1.8),
        ("middle east", 1.8), ("ukraine", 2.0), ("russia", 1.6), ("retaliat", 1.8),
        ("export curb", 2.0), ("export control", 2.0), ("border", 1.2),
        ("ceasefire", 2.2), ("military strike", 2.6), ("terror", 2.2),
    ],
    EventClass.MACROECONOMIC: [
        ("inflation", 2.5), ("federal reserve", 3.5), ("fed", 2.0),
        ("interest rate", 3.0), ("rate hike", 2.5), ("rate cut", 2.5),
        ("cpi", 2.6), ("gdp", 2.6), ("unemployment", 2.4), ("recession", 3.0),
        ("jobs report", 2.6), ("payrolls", 2.4), ("rate decision", 3.0),
        ("powell", 2.4), ("quantitative", 2.4), ("treasury yield", 2.4),
        ("central bank", 2.8), ("economy", 1.6), ("economic", 1.4),
        ("stimulus", 2.2), ("consumer spending", 1.8), ("dovish", 2.4),
        ("hawkish", 2.4), ("soft landing", 2.2), ("monetary policy", 2.8),
    ],
    EventClass.CREDIT_EVENT: [
        ("default", 3.2), ("bankrupt", 3.6), ("insolven", 3.2), ("credit rating", 3.0),
        ("credit downgrade", 3.2), ("debt restructuring", 3.2), ("missed payment", 3.2),
        ("bondholder", 2.2), ("junk bond", 2.6), ("arrears", 3.2), ("credit event", 3.2),
        ("covenant breach", 3.2), ("distress", 2.6), ("nonperforming", 2.8),
        ("non-performing", 2.8), ("downgrad", 2.0), ("liquidity crisis", 3.2),
        ("unable to repay", 3.2), ("standstill", 2.6),
    ],
    EventClass.MERGER_ACQUISITION: [
        ("merger", 3.2), ("acqui", 3.0), ("takeover", 3.2), ("buyout", 3.2),
        ("bid for", 2.6), ("to buy", 2.4), ("combines", 2.0), ("combination", 1.8),
        ("spin-off", 2.4), ("spinoff", 2.4), ("spin off", 2.4), ("consolidat", 2.0),
        ("tender offer", 3.2), ("all-stock", 2.4), ("all-cash", 2.2),
        ("stake in", 1.8), ("agreed to buy", 3.0), ("acquisition", 3.0),
    ],
    EventClass.PRODUCT_LAUNCH: [
        ("launch", 2.6), ("launches", 2.6), ("launched", 2.6), ("unveil", 2.6),
        ("debut", 2.2), ("rolls out", 2.6), ("rollout", 2.2), ("roll-out", 2.2),
        ("new product", 2.2), ("introduc", 2.0), ("next-gen", 2.2),
        ("next generation", 2.2), ("beta", 1.6), ("announced a new", 2.2),
        ("release", 1.6), ("preorder", 1.8), ("pre-order", 1.8),
    ],
    EventClass.EARNINGS: [
        ("earnings", 3.2), ("quarterly result", 3.2), ("eps", 3.0), ("revenue", 2.2),
        ("profit warning", 3.2), ("guidance", 2.2), ("forecast", 1.8), ("forecasts", 1.8),
        ("estimate", 1.6), ("net income", 2.6), ("results beat", 3.2),
        ("results miss", 3.2), ("same-store sales", 2.6), ("margin", 1.6),
        ("topline", 2.0), ("bottom line", 2.0), ("quarter", 1.2), ("q1", 1.0),
        ("q2", 1.0), ("q3", 1.0), ("q4", 1.0), ("full-year", 1.6), ("sales rose", 2.4),
        ("comp sales", 2.4), ("dividend", 1.4), ("buyback", 1.6), ("payout", 1.6),
    ],
    EventClass.REGULATORY: [
        ("antitrust", 3.2), ("sec ", 2.0), ("fda", 2.6), ("regulator", 2.2),
        ("regulatory", 2.2), ("investigation", 2.4), ("probe", 2.4),
        ("lawsuit", 2.4), ("litigation", 2.6), ("fine", 1.8), ("fined", 2.4),
        ("penalty", 2.2), ("ruling", 2.2), ("compliance", 1.8), ("subpoena", 2.8),
        ("class action", 2.8), ("ban", 1.8), ("approval", 1.6), ("license", 1.6),
        ("rule", 1.2), ("settlement", 2.2), ("settle", 2.0), ("court", 1.8),
        ("court order", 2.4), ("monopoly", 2.6), ("privacy", 1.8),
    ],
    EventClass.OPERATIONAL: [
        ("recall", 3.2), ("outage", 3.0), ("breach", 3.0), ("hack", 3.0),
        ("cyber", 2.6), ("strike", 2.6), ("supply chain", 2.6), ("factory", 1.6),
        ("delay", 2.0), ("delays", 2.0), ("halt", 2.2), ("suspend", 2.2),
        ("shutdown", 2.6), ("layoff", 2.6), ("disrupt", 2.2), ("disruption", 2.2),
        ("shortage", 2.2), ("defect", 2.6), ("glitch", 2.2), ("outages", 3.0),
    ],
}

# ------------------------------------------------------------------------- impact
IMPACT_BASE: dict[EventClass, float] = {
    EventClass.GEOPOLITICAL: 6.0,
    EventClass.MACROECONOMIC: 6.0,
    EventClass.CREDIT_EVENT: 6.5,
    EventClass.MERGER_ACQUISITION: 5.0,
    EventClass.PRODUCT_LAUNCH: 4.0,
    EventClass.EARNINGS: 5.0,
    EventClass.REGULATORY: 5.0,
    EventClass.OPERATIONAL: 5.0,
    EventClass.OTHER: 3.0,
}

# words that push severity regardless of class
INTENSITY_STEMS = [
    "halt", "collaps", "bankrupt", "default", "breach", "recall", "sanction",
    "probe", "investigat", "fraud", "scandal", "crisis", "plunge", "suspend",
    "strike", "layoff", "downgrad", "insolven", "miss", "warn", "surge",
    "record", "all-time", "outage", "hack", "lawsuit", "ban", "recession",
    "collapse", "crash", "fail", "breach", "curb",
]


# ---------------------------------------------------------------------------
# Additions found by error analysis on Financial PhraseBank (scripts/evaluate.py):
# plain direction words that carry the sentiment of reported-results sentences
# ("net sales increased", "profit decreased"), contract wins, and shrinking losses.
# ---------------------------------------------------------------------------
SENTIMENT_LEXICON.update({
    "increase": 1.2, "increases": 1.2, "increased": 1.4, "increasing": 1.2,
    "decrease": -1.2, "decreases": -1.2, "decreased": -1.4, "decreasing": -1.2,
    "down": -1.0, "signed": 1.0, "awarded": 1.4,
    "strengthen": 1.3, "strengthens": 1.3, "strengthened": 1.3,
    "temporarily": -0.8,
})
PHRASE_LEXICON.update({
    "loss narrowed": 1.8, "loss narrows": 1.8, "losses narrowed": 1.8,
    "loss shrank": 1.8, "loss decreased": 1.8, "losses decreased": 1.8,
    "loss decreases": 1.8,
})
