"""The answer call: short queries, and surviving a dropped connection.

Both come from the 223-question campaign rather than from imagination.
A bare "?" produced an answer referring to "the episode title you're
asking about" when no title had been asked about, and one call in ~320
died on a TimeoutError after the expensive embedding and rerank work was
already paid for.

Neither is dangerous in a CLI. Both are a published reply on a bot.
"""

import asyncio

import anthropic
import pytest

from app.search import (
    ANSWER_RETRIES,
    EMPTY_QUERY_ANSWER,
    MIN_QUERY_CHARS,
    MCGIndex,
)

# Captured before anything can rebind the name. Patching asyncio.sleep
# with a lambda that calls asyncio.sleep recurses forever, because by the
# time the lambda runs the name already points at the lambda.
_REAL_SLEEP = asyncio.sleep


@pytest.fixture
def nosleep(monkeypatch):
    """Skip the retry backoff so the suite stays fast."""
    async def _instant(*_args, **_kw):
        await _REAL_SLEEP(0)
    monkeypatch.setattr(asyncio, "sleep", _instant)


class Boom(Exception):
    """A non-transport failure. Must not be retried."""


def _index() -> MCGIndex:
    return MCGIndex()


# --- short queries: answered without spending anything -----------------

@pytest.mark.parametrize("query", ["", " ", "?", "??", "a", " a ", "\n"])
def test_a_query_with_no_subject_is_turned_away(query):
    """And crucially, without embedding it or calling the model.

    If this ever regresses to a real search, the tell is that these
    become slow — a query too short to carry meaning retrieves whatever
    is nearest and asks the model to explain it.
    """
    idx = _index()
    idx.retrieve = None          # any use of retrieval would raise here
    r = asyncio.run(idx.search(query))
    assert r.answer == EMPTY_QUERY_ANSWER
    assert r.hits == []
    assert not r.refused         # not a refusal — nothing was asked


@pytest.mark.parametrize("query", ["$UP", "wen", "gm?", "$WTH"])
def test_short_but_real_questions_still_go_through(query):
    """Tickers and crypto-X shorthand are genuine questions.

    "$UP" is a real project in this archive and "wen" is a whole question
    on X, so the floor has to sit below them.
    """
    assert len(query.strip()) >= MIN_QUERY_CHARS


# --- the answer call retries transport faults only ---------------------

def _flaky(fail_times: int, exc: Exception):
    """A messages.create that fails `fail_times` then succeeds."""
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc
        return type("R", (), {"stop_reason": "end_turn", "model": "m",
                              "content": []})()
    return create, calls


def _patch(idx, create):
    class Msgs:
        pass
    msgs = Msgs()
    msgs.create = create

    class Beta:
        pass
    beta = Beta()
    beta.messages = msgs

    class Client:
        def with_options(self, **_):
            return self
    client = Client()
    client.beta = beta
    idx._anthropic = client
    return idx


@pytest.mark.parametrize("exc", [
    anthropic.APITimeoutError(request=None),
    TimeoutError(),
])
def test_a_dropped_connection_is_retried_and_succeeds(exc, nosleep):
    create, calls = _flaky(1, exc)
    idx = _patch(_index(), create)
    asyncio.run(idx._answer("q", []))
    assert calls["n"] == 2       # failed once, retried once, succeeded


def test_it_gives_up_rather_than_retrying_forever(nosleep):
    create, calls = _flaky(99, anthropic.APITimeoutError(request=None))
    idx = _patch(_index(), create)
    with pytest.raises(anthropic.APITimeoutError):
        asyncio.run(idx._answer("q", []))
    assert calls["n"] == ANSWER_RETRIES


def test_a_real_error_is_not_retried(nosleep):
    """A bad request or an auth failure is a true answer about the
    request. Repeating it spends money to fail the same way, and on a
    proxy billed per call that is a real cost."""
    create, calls = _flaky(99, Boom("bad request"))
    idx = _patch(_index(), create)
    with pytest.raises(Boom):
        asyncio.run(idx._answer("q", []))
    assert calls["n"] == 1


def test_a_first_attempt_that_works_costs_one_call():
    create, calls = _flaky(0, Boom("unused"))
    idx = _patch(_index(), create)
    asyncio.run(idx._answer("q", []))
    assert calls["n"] == 1
