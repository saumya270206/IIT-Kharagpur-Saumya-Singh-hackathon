"""NLP engine: sentiment, event classification, impact, entity resolution."""

import pytest

from risk_engine.models import EventClass, RawItem, RiskSignal
from risk_engine.nlp.local import (
    LocalAnalyzer, classify_event, resolve_entity, score_impact, score_sentiment,
)


@pytest.mark.parametrize("text", [
    "Shares surge on record profits and strong growth",
    "The launch was a huge success with breakthrough results",
])
def test_positive_sentiment(text):
    assert score_sentiment(text) > 0.3


@pytest.mark.parametrize("text", [
    "Shares plunge after a fraud probe and bankruptcy risk",
    "The outage caused a massive selloff and angry customers",
])
def test_negative_sentiment(text):
    assert score_sentiment(text) < -0.3


def test_neutral_sentiment():
    assert score_sentiment("The company will hold its annual meeting on Tuesday") == 0.0


def test_negation_flips_polarity():
    assert score_sentiment("Profits did not rise this quarter") < \
        score_sentiment("Profits rise this quarter")


def test_sentiment_bounds():
    for text in ("amazing amazing amazing record surge boom!!!",
                 "catastrophic collapse fraud bankruptcy default crash plunge!!!", ""):
        assert -1.0 <= score_sentiment(text) <= 1.0


@pytest.mark.parametrize("text,expected", [
    ("War and sanctions escalate as border conflict worsens", EventClass.GEOPOLITICAL),
    ("The Federal Reserve signaled a rate cut as inflation cooled", EventClass.MACROECONOMIC),
    ("Moody's downgraded the bank after a missed debt payment", EventClass.CREDIT_EVENT),
    ("Microsoft agreed to acquire the startup in a merger deal", EventClass.MERGER_ACQUISITION),
    ("The company unveiled its new smartphone launch event", EventClass.PRODUCT_LAUNCH),
    ("Quarterly earnings beat estimates with higher revenue", EventClass.EARNINGS),
    ("EU opens antitrust investigation and imposes a fine", EventClass.REGULATORY),
    ("Data breach and outage knock services offline after hack", EventClass.OPERATIONAL),
])
def test_event_classification(text, expected):
    assert classify_event(text)[0] is expected


def test_unknown_event_is_other():
    cls, conf = classify_event("The cat sat quietly on the warm windowsill")
    assert cls is EventClass.OTHER and conf < 0.5


def test_impact_bounds_and_severity_ordering():
    mild = "Company opens a new office"
    severe = "BREAKING: massive fraud, bankruptcy and default trigger a historic crash!"
    s_mild = score_impact(mild, score_sentiment(mild), *classify_event(mild))
    s_sev = score_impact(severe, score_sentiment(severe), *classify_event(severe))
    assert 1 <= s_mild <= 10 and 1 <= s_sev <= 10
    assert s_sev > s_mild


def test_cashtag_and_name_resolution():
    assert resolve_entity("$NVDA to the moon", None, None)[0] == "NVDA"
    assert resolve_entity("Tesla recalls vehicles", None, None)[0] == "TSLA"
    assert resolve_entity("Bank of America cuts guidance", None, None)[0] == "BAC"
    assert resolve_entity("random text", "msft", "Microsoft") == ("MSFT", "Microsoft")
    assert resolve_entity("nothing relevant here", None, None) == (None, None)


def test_full_signal_shape():
    item = RawItem(id="t1", source="news",
                   text="JPMorgan misses estimates, credit downgrade sparks selloff")
    sig = LocalAnalyzer().analyze(item)
    assert isinstance(sig, RiskSignal)
    assert sig.ticker == "JPM"
    assert sig.event_class in (EventClass.CREDIT_EVENT, EventClass.EARNINGS)
    assert sig.sentiment < 0 and 1 <= sig.impact <= 10
    assert set(sig.to_json()) >= {"sentiment", "event_class", "impact", "ticker"}


@pytest.mark.parametrize("text", [
    "Bank of America reports solid results",     # 'ban' must not match 'bank'
    "Coupon payments rise",                      # 'coup' must not match 'coupon'
    "The cat sat quietly on the warm windowsill",  # 'war' must not match 'warm'
    "Warning issued over warehouse fire",        # ... nor 'warning' / 'warehouse'
    "Financial markets were calm",               # 'fine' must not match 'financial'
])
def test_short_stems_do_not_match_inside_words(text):
    assert classify_event(text)[0] is EventClass.OTHER


def test_short_stems_still_match_inflections():
    assert classify_event("US banned chip exports")[0] is EventClass.REGULATORY
    assert classify_event("Warring factions clash")[0] is EventClass.GEOPOLITICAL


def test_rising_risk_quantities_are_negative_and_falling_are_positive():
    # "surges"/"spike" are positive words on their own; domain phrases override them
    assert score_sentiment("Fed signals rate hikes as inflation surges and yields spike") < -0.5
    assert score_sentiment("Losses widen as defaults climb") < -0.5
    assert score_sentiment("Inflation cooled as the Fed signaled a rate cut") > 0.4
    assert score_sentiment("Sales surge on strong demand") > 0.5   # generic usage unchanged
