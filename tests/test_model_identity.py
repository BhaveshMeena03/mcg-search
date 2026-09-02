"""Deciding whether the model that answered is the model we asked for.

This is the single most important check against an inference proxy: the
whole proposition is "same weights, cheaper", and a proxy quietly routing
to something smaller would look like a bargain and be a downgrade.

It has to thread a needle. Too strict and every naming convention reads
as a substitution — Anthropic returns a dated snapshot, usepod returns a
vendor-prefixed dotted id, and neither is a different model. Too loose
and a genuine downgrade slips through, which is the failure that costs
something.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_usepod_model", ROOT / "scripts" / "check_usepod.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("served", [
    "claude-haiku-4-5",                    # exact
    "claude-haiku-4-5-20251001",           # Anthropic's dated snapshot
    "anthropic/claude-haiku-4.5",          # usepod's marketplace id
    "anthropic/claude-haiku-4-5",          # prefixed, dashed
    "Anthropic/Claude-Haiku-4.5",          # case
])
def test_accepts_the_same_model_under_any_naming_scheme(served):
    mod = _load()
    assert mod._same_model(served, "claude-haiku-4-5")


@pytest.mark.parametrize("served", [
    "claude-haiku-3",
    "claude-3-haiku",
    "anthropic/claude-sonnet-4.5",
    "claude-sonnet-5",
    "gpt-oss-120b",
    "qwen3-32b",
    "",
])
def test_rejects_a_different_model(served):
    """The failure worth catching: something cheaper served silently."""
    mod = _load()
    assert not mod._same_model(served, "claude-haiku-4-5")


def test_a_smaller_haiku_is_not_the_haiku_we_asked_for():
    """Guards the prefix match specifically.

    'claude-haiku-4' must not satisfy a request for 'claude-haiku-4-5',
    and the reverse direction must not accidentally pass either.
    """
    mod = _load()
    assert not mod._same_model("claude-haiku-4", "claude-haiku-4-5")
    assert mod._same_model("claude-haiku-4-5-20251001", "claude-haiku-4-5")


def test_normalisation_is_narrow():
    mod = _load()
    assert mod._normalise_model("anthropic/claude-haiku-4.5") == \
        "claude-haiku-4-5"
    assert mod._normalise_model("") == ""
