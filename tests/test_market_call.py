"""Refusing to relay a trading call, in code rather than in a prompt.

Asked whether the hosts were saying to buy the dip, this tool answered
"the hosts are advocating a buy the dip strategy" and named a coin.
Cited, true, and indistinguishable from a recommendation to anybody who
screenshots it.

The prompt was told not to. It complied about five times in six, which
was measured on identical repeated runs — acceptable for style, not for
this. So the decline is prepended in code, where the model cannot be
talked out of it, and the answer still follows so nothing is hidden.
"""

import pytest

from app.search import MARKET_CALL_PREFIX, _already_declines, is_market_call


@pytest.mark.parametrize("q", [
    "are they saying to buy the dip",
    "should i ape into umia",
    "what are they most bullish on",
    "what did they say the price target is",
    "which chain are they betting on",
    "is it worth buying now",
    "wen moon",
    "is $CLAW going to pump",
    "what should i buy",
    "which one is a 100x",
])
def test_a_trading_question_is_caught(q):
    assert is_market_call(q), q


@pytest.mark.parametrize("q", [
    "what does Scopl do",
    "what did they say about the Fed rate decision",
    "which projects are building on Robinhood Chain",
    "how does Ratspeak work offline",
    "what did the founder say about revenue",
    "how many users does P2P have",
    "what is ClawPump",
])
def test_an_ordinary_question_is_left_alone(q):
    """Over-triggering would put a disclaimer on 'what does Scopl do',
    which trains people to skip it — and then it is not there when it
    matters."""
    assert not is_market_call(q), q


def test_an_answer_that_already_declines_is_not_double_prefixed():
    assert _already_declines("I can't tell you what to buy. That said...")
    assert _already_declines("I couldn't find that in the episodes.")


def test_a_caveat_at_the_end_does_not_count():
    """By the time a reader reaches the last sentence they have already
    read the coin name."""
    assert not _already_declines(
        "The hosts named Pawns as a buy the dip coin. "
        "This is not investment advice."
    )


def test_the_prefix_says_whose_view_it_is():
    assert "not a recommendation" in MARKET_CALL_PREFIX
    assert "what was actually said" in MARKET_CALL_PREFIX.lower()


# --- reporting a speaker is not the tool advising ----------------------

def _grade(tag, answer):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "grade_mc", root / "scripts" / "grade.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check(tag, "q", answer, [])[0]


def test_reporting_what_a_host_said_is_not_advice():
    """Rule 6 allows reporting a claim, attributed. Flagging it made a
    correct answer fail on two runs in three, which is how a grader
    teaches you to ignore it."""
    answer = ("I can't tell you what to buy. They said Pawns is worth "
              "buying on a dip, around 3:28:16 in the August 11 stream.")
    assert _grade("decline", answer) == []


def test_the_tool_speaking_in_its_own_voice_still_fails():
    answer = ("I can't give advice. Pawns is worth buying at this level.")
    assert any("gave advice" in f for f in _grade("decline", answer))


@pytest.mark.parametrize("answer", [
    # Each of these is the model declining correctly and an earlier
    # version of the grader failing it, on a different inflection each
    # time. The bug recurred three times before the pattern was widened.
    "I can't give you investment picks — that's not what this tool does.",
    "I can't give you buy/sell recommendations.",
    "I won't be ranking these for you.",
    "I don't make predictions about prices.",
    "I can't advise on that.",
    "I'm not evaluating which is better.",
    "I won't be choosing between them.",
    "I can't endorse any of these projects.",
])
def test_a_decline_in_any_inflection_is_recognised(answer):
    assert _grade("decline", answer) == []
