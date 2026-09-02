"""Credential redaction in the proxy checker.

usepod puts its auth token in the URL PATH rather than a header. That
means the token rides inside every URL the SDK builds, so it appears in
connection errors, timeouts and stack traces — places an API key in a
header never reaches. Printing an SDK exception raw would put a live
credential into terminal scrollback and CI logs.

These tests exist because that failure is invisible when it happens: the
script still works, the output just quietly contains a secret.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_usepod", ROOT / "scripts" / "check_usepod.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_redacts_a_registered_secret():
    mod = _load()
    mod._SECRETS.append("tok_abcdef123456")
    assert "tok_abcdef123456" not in mod._redact(
        "GET https://api.usepod.ai/proxy/tok_abcdef123456/v1/messages failed"
    )


def test_redacts_inside_an_exception_message():
    """The realistic case: the token arrives embedded in an SDK error."""
    mod = _load()
    mod._SECRETS.append("tok_abcdef123456")
    exc = ConnectionError(
        "Cannot connect to https://api.usepod.ai/proxy/tok_abcdef123456"
    )
    assert "tok_abcdef123456" not in mod._exc(exc)
    assert "ConnectionError" in mod._exc(exc)


def test_redacts_before_truncating():
    """Order matters.

    Truncating a long error first can slice a token in half and print
    the first half, which redaction then no longer matches because the
    full string is gone.
    """
    mod = _load()
    token = "tok_" + "z" * 40
    mod._SECRETS.append(token)
    # Pad so the token straddles the 220-character truncation point.
    exc = RuntimeError("x" * 200 + token + " trailing detail")
    out = mod._exc(exc)
    assert token not in out
    assert token[:20] not in out          # not even a usable fragment


def test_short_strings_are_not_treated_as_secrets():
    """A 'secret' of a few characters would redact ordinary words.

    An empty or tiny value must not turn every message into <redacted>,
    which would hide the errors this script exists to surface.
    """
    mod = _load()
    mod._SECRETS.append("v1")
    assert mod._redact("connecting to /v1/messages") == "connecting to /v1/messages"


def test_nothing_registered_is_a_passthrough():
    mod = _load()
    mod._SECRETS.clear()
    assert mod._redact("plain message") == "plain message"


def test_extracts_the_token_from_an_anthropic_proxy_url():
    mod = _load()
    assert mod._token_from(
        "https://api.usepod.ai/proxy/tok_abcdef123456"
    ) == "tok_abcdef123456"
    # A trailing slash must not yield an empty secret.
    assert mod._token_from(
        "https://api.usepod.ai/proxy/tok_abcdef123456/"
    ) == "tok_abcdef123456"


def test_does_not_mistake_a_path_segment_for_a_token():
    """An OpenAI-style URL ends in /v1, and a bare host has no token.

    Registering either as a secret would strip ordinary words out of
    every message the script prints, hiding the errors it exists to show.
    """
    mod = _load()
    assert mod._token_from("https://api.usepod.ai/proxy/tok_real/v1") == ""
    assert mod._token_from("https://api.usepod.ai") == ""
    assert mod._token_from("https://api.anthropic.com") == ""
    assert mod._token_from("") == ""
