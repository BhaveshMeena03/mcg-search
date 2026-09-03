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
    # Isolate the retry loop. The failover is exercised separately below;
    # leaving a real fallback client here would send the retry down a
    # network call instead of back through the stub.
    idx._fallback = None
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


# --- failing over to Anthropic when the proxy dies ---------------------

def _with_fallback(idx, proxy_create, direct_create):
    """A proxy client and a direct one, both stubbed."""
    def wrap(create):
        msgs = type("M", (), {})()
        msgs.create = create
        beta = type("B", (), {})()
        beta.messages = msgs
        c = type("C", (), {"with_options": lambda self, **_: self})()
        c.beta = beta
        return c
    idx._anthropic = wrap(proxy_create)
    idx._fallback = wrap(direct_create)
    idx._proxy_failed_at = 0.0
    return idx


def test_a_dead_proxy_falls_over_to_anthropic(nosleep):
    """The measured failure: the route moved to a relay that stopped
    answering. Nothing in this repo caused it or could prevent it, so the
    only defence is a way back."""
    proxy, pcalls = _flaky(99, anthropic.APITimeoutError(request=None))
    direct, dcalls = _flaky(0, Boom("unused"))
    idx = _with_fallback(_index(), proxy, direct)
    asyncio.run(idx._answer("q", []))
    assert pcalls["n"] == 1      # tried the proxy once
    assert dcalls["n"] == 1      # then answered on Anthropic


def test_the_proxy_is_skipped_for_the_cooldown(nosleep):
    """An outage should cost one slow answer, not one per visitor."""
    from app.search import PROXY_COOLDOWN_SECONDS
    proxy, pcalls = _flaky(99, anthropic.APITimeoutError(request=None))
    direct, dcalls = _flaky(0, Boom("unused"))
    idx = _with_fallback(_index(), proxy, direct)
    asyncio.run(idx._answer("q", []))
    for _ in range(3):
        asyncio.run(idx._answer("q", []))
    assert pcalls["n"] == 1              # never retried inside the window
    assert dcalls["n"] == 4
    assert PROXY_COOLDOWN_SECONDS == 300


def test_the_proxy_is_tried_again_once_the_cooldown_expires(nosleep):
    import time as _time

    from app.search import PROXY_COOLDOWN_SECONDS
    proxy, pcalls = _flaky(1, anthropic.APITimeoutError(request=None))
    direct, dcalls = _flaky(0, Boom("unused"))
    idx = _with_fallback(_index(), proxy, direct)
    asyncio.run(idx._answer("q", []))
    idx._proxy_failed_at = _time.monotonic() - PROXY_COOLDOWN_SECONDS - 1
    asyncio.run(idx._answer("q", []))
    assert pcalls["n"] == 2              # recovery needs no deploy


def test_the_fallback_never_points_at_the_proxy():
    """The subtle one, and the reason base_url is named explicitly.

    The SDK reads ANTHROPIC_BASE_URL from the environment when it is not
    told otherwise, which is exactly how the proxy is configured. A
    fallback built with only a key inherits the proxy and quietly becomes
    a second route to the thing that just failed.
    """
    from app.search import ANTHROPIC_DIRECT_URL
    idx = _index()
    if idx._fallback is not None:
        assert str(idx._fallback.base_url).rstrip("/") == ANTHROPIC_DIRECT_URL


def test_a_proxy_token_is_redacted_from_logs():
    from app.search import _redact_proxy
    msg = "Connection to https://api.usepod.ai/proxy/tok_secret123/v1 failed"
    assert "tok_secret123" not in _redact_proxy(msg)
    assert "<token>" in _redact_proxy(msg)
