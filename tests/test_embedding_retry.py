"""Retrying transient embedding failures.

Written after a 54-question run lost 52 of them to
"APIConnectionError: Error communicating with Voyage" during a few
seconds of DNS trouble. Only RateLimitError was retried, so every
question in flight died. On a public bot that is every mention failing
for as long as the wobble lasts.

The tests patch asyncio.sleep so backoff is asserted, not waited for.
"""

import asyncio

import pytest
from voyageai import error as voyage_error

from app import embeddings


class FakeResult:
    def __init__(self, n):
        self.embeddings = [[0.1] * 4 for _ in range(n)]


class FakeClient:
    """Fails `failures` times with `exc`, then succeeds."""

    def __init__(self, exc, failures):
        self.exc, self.failures, self.calls = exc, failures, 0

    async def embed(self, texts, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return FakeResult(len(texts))


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return slept


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _embed(client):
    return await embeddings.embed_texts(
        client, ["hello"], model="voyage-3.5", dimension=4,
        input_type="query",
    )


@pytest.mark.parametrize("exc", [
    voyage_error.APIConnectionError("Error communicating with Voyage"),
    voyage_error.Timeout("timed out"),
    voyage_error.ServiceUnavailableError("503"),
    voyage_error.ServerError("500"),
])
def test_retries_transient_failures(exc):
    """The real case. One blip must cost a pause, not the answer."""
    client = FakeClient(exc, failures=2)
    out = asyncio.run(_embed(client))
    assert len(out) == 1
    assert client.calls == 3


def test_still_retries_rate_limits():
    client = FakeClient(voyage_error.RateLimitError("429"), failures=1)
    assert len(asyncio.run(_embed(client))) == 1
    assert client.calls == 2


def test_gives_up_eventually_rather_than_hanging():
    client = FakeClient(voyage_error.APIConnectionError("down"), failures=99)
    with pytest.raises(voyage_error.APIConnectionError):
        asyncio.run(_embed(client))
    assert client.calls == embeddings.MAX_RETRIES


def test_does_not_retry_a_bad_key():
    """A wrong key fails identically six times. Retrying only delays the
    error by a minute and hides which failure it was."""
    client = FakeClient(voyage_error.AuthenticationError("bad key"), failures=99)
    with pytest.raises(voyage_error.AuthenticationError):
        asyncio.run(_embed(client))
    assert client.calls == 1


def test_does_not_retry_a_malformed_request():
    client = FakeClient(voyage_error.InvalidRequestError("bad input"), failures=99)
    with pytest.raises(voyage_error.InvalidRequestError):
        asyncio.run(_embed(client))
    assert client.calls == 1


def test_connection_backoff_is_shorter_than_a_rate_limit_backoff(no_real_sleeping):
    """A rate limit needs the window to pass; a dropped socket needs a
    second. Starting at 25s for a blip would make a bot feel broken."""
    asyncio.run(_embed(FakeClient(voyage_error.APIConnectionError("x"),
                                  failures=1)))
    conn_waits = list(no_real_sleeping)
    no_real_sleeping.clear()
    asyncio.run(_embed(FakeClient(voyage_error.RateLimitError("x"),
                                  failures=1)))
    assert conn_waits[0] < no_real_sleeping[0]
    assert conn_waits[0] <= 2
