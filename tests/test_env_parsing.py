"""Reading values out of .env.

These exist because of a real 10 minutes lost. usepod's dashboard hands
you shell-export lines:

    export ANTHROPIC_API_KEY="UsePod"
    export ANTHROPIC_BASE_URL=https://api.usepod.ai/proxy/<token>

Pasted straight into .env — the obvious thing to do — a parser matching
only a bare KEY=value reads nothing, and then reports the base URL as
missing while it is sitting in the file three lines up. A config parser
that silently returns "" for a value that is present is worse than one
that throws.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Load the script with ROOT pointed at a temp dir holding a .env."""
    def _write(contents: str):
        (tmp_path / ".env").write_text(contents)
        spec = importlib.util.spec_from_file_location(
            "check_usepod_env", ROOT / "scripts" / "check_usepod.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = tmp_path
        return module
    return _write


def test_plain_key_value(env):
    mod = env("ANTHROPIC_BASE_URL=https://api.usepod.ai/proxy/tok123\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == \
        "https://api.usepod.ai/proxy/tok123"


def test_export_prefix(env):
    """The form usepod's site actually gives you."""
    mod = env("export ANTHROPIC_BASE_URL=https://api.usepod.ai/proxy/tok123\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == \
        "https://api.usepod.ai/proxy/tok123"


def test_export_with_double_quotes(env):
    mod = env('export ANTHROPIC_API_KEY="UsePod"\n')
    assert mod._from_env_file("ANTHROPIC_API_KEY") == "UsePod"


def test_export_with_single_quotes(env):
    mod = env("export ANTHROPIC_API_KEY='UsePod'\n")
    assert mod._from_env_file("ANTHROPIC_API_KEY") == "UsePod"


def test_leading_whitespace(env):
    mod = env("   export ANTHROPIC_BASE_URL=https://x/proxy/tok\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == "https://x/proxy/tok"


def test_missing_key_is_empty_not_an_error(env):
    mod = env("OTHER=1\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == ""


def test_empty_value_reads_as_empty(env):
    """The exact paste that started this: the URL with no token after it."""
    mod = env("export ANTHROPIC_BASE_URL=\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == ""


def test_does_not_match_a_key_that_merely_shares_a_prefix(env):
    mod = env("ANTHROPIC_BASE_URL_OLD=https://stale\n"
              "ANTHROPIC_BASE_URL=https://current\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == "https://current"


def test_a_commented_line_is_not_a_value(env):
    """The template line shipped in .env is commented out.

    Reading it would hand back the literal string PASTE_TOKEN_HERE and
    produce a confusing 401 instead of 'not configured'.
    """
    mod = env("#ANTHROPIC_BASE_URL=https://api.usepod.ai/proxy/PASTE_TOKEN_HERE\n")
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == ""


def test_no_env_file_at_all(env, tmp_path):
    mod = env("PLACEHOLDER=1\n")
    (tmp_path / ".env").unlink()
    assert mod._from_env_file("ANTHROPIC_BASE_URL") == ""
