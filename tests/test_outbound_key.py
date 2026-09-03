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


# --- refusing to start on a config that cannot answer -------------------

def test_direct_without_a_key_is_refused_at_startup():
    """The real failure: the shell exported ANTHROPIC_BASE_URL pointing
    at Anthropic while the key had been removed. Every request raised an
    SDK TypeError about authentication methods and the page just said
    the answer stopped early."""
    import pytest

    from app.search import Misconfigured, check_credentials

    class S:
        anthropic_api_key = ""
        anthropic_base_url = "https://api.anthropic.com"

    with pytest.raises(Misconfigured) as exc:
        check_credentials(S())
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    # The cause is almost always a shell variable beating .env, so say so.
    assert "shell" in str(exc.value).lower()


def test_no_base_url_and_no_key_is_also_refused():
    import pytest

    from app.search import Misconfigured, check_credentials

    class S:
        anthropic_api_key = ""
        anthropic_base_url = None

    with pytest.raises(Misconfigured):
        check_credentials(S())


def test_a_proxy_without_an_anthropic_key_is_fine():
    """The intended production shape: the proxy authenticates on its own
    URL token, so no Anthropic credential is needed or wanted."""
    from app.search import check_credentials

    class S:
        anthropic_api_key = ""
        anthropic_base_url = "https://api.usepod.ai/proxy/tok"

    check_credentials(S())        # must not raise
