"""Telling apart the reasons a fetch failed.

These look identical from outside — no transcript — and need opposite
responses. A block means stop and get cookies or wait. No captions means
this episode will never work and re-running is pointless.

Across the full 407-episode archive exactly one episode had no captions,
and the diagnostic reported "(no stderr)", which reads like a fault
worth re-investigating rather than a permanent, known gap.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "fetch_episodes", ROOT / "scripts" / "fetch_episodes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stderr", [
    "ERROR: Sign in to confirm you're not a bot",
    "ERROR: unable to download video subtitles: HTTP Error 429: Too Many Requests",
    "ERROR: HTTP Error 403: Forbidden",
])
def test_names_a_block_as_a_block(stderr):
    """Actionable: needs cookies, a proxy, or waiting."""
    assert "BLOCKED" in _load()._diagnose(stderr)


def test_names_missing_captions_explicitly():
    out = _load()._diagnose("ERROR: no automatic captions found for video")
    assert "no captions" in out
    assert "BLOCKED" not in out


def test_silent_exit_is_reported_as_missing_captions_not_as_nothing():
    """The real case from the 407-episode run.

    yt-dlp writes no file and no error when a video simply has none.
    "(no stderr)" made a permanent gap look like a transient fault.
    """
    out = _load()._diagnose("")
    assert "no captions available" in out
    assert "list-subs" in out          # tells you how to confirm it
    assert "no stderr" not in out


def test_an_unrecognised_error_is_passed_through_not_swallowed():
    out = _load()._diagnose("ERROR: something nobody anticipated happened")
    assert "nobody anticipated" in out


def test_a_block_wins_over_a_caption_message():
    """A rate-limited request can mention both. The block is the
    actionable half, and treating it as 'no captions' would wrongly mark
    a fetchable episode as permanently impossible."""
    out = _load()._diagnose(
        "ERROR: no subtitles found\nERROR: HTTP Error 429: Too Many Requests"
    )
    assert "BLOCKED" in out
