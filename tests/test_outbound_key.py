"""Where the real Anthropic key is allowed to go.

Routing through an inference proxy is advertised as a base-URL swap, and
the obvious implementation of that forwards the configured api_key to
whatever host the URL now points at — shipping a live sk-ant-... to a
third party on every request. Proxies that authenticate on a path token
ignore the key anyway, so there is nothing to gain and a credential to
lose.

This is the kind of leak that never announces itself: everything works,
answers come back, and the key is simply also somewhere else now.
"""

from app.search import PLACEHOLDER_KEY, _outbound_key

REAL = "sk-ant-real-credential"


class Settings:
    def __init__(self, base_url):
        self.anthropic_api_key = REAL
        self.anthropic_base_url = base_url


def test_real_key_when_going_direct_to_anthropic():
    assert _outbound_key(Settings(None)) == REAL
    assert _outbound_key(Settings("")) == REAL
    assert _outbound_key(Settings("https://api.anthropic.com")) == REAL


def test_placeholder_for_a_third_party_proxy():
    s = Settings("https://api.usepod.ai/proxy/tok_abc123")
    assert _outbound_key(s) == PLACEHOLDER_KEY
    assert REAL not in _outbound_key(s)


def test_placeholder_for_any_unknown_host():
    """Default deny. A new proxy must not inherit the credential by
    virtue of nobody having thought about it yet."""
    for url in ("https://openrouter.ai/api",
                "https://some-relay.example.com/v1",
                "http://localhost:8080"):
        assert _outbound_key(Settings(url)) == PLACEHOLDER_KEY


def test_whitespace_only_base_url_counts_as_unset():
    assert _outbound_key(Settings("   ")) == REAL


def test_a_lookalike_host_does_not_get_the_key():
    """api.anthropic.com.evil.example is not Anthropic.

    Substring matching is what makes this test necessary — the check
    must not be fooled by a hostname that merely contains the real one
    as a prefix of a longer domain.
    """
    s = Settings("https://api.anthropic.com.evil.example/v1")
    assert _outbound_key(s) == PLACEHOLDER_KEY
