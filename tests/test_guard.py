"""The guard that keeps this repo away from the live Bullpen index.

This is the one test in the suite that protects something outside this
repo. search.lexthedev.com reads index 'bullpen-concierge', namespace
'podcast'; a wrong .env here is the only path from this codebase to that
data, so the refusal is asserted rather than assumed.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_ingest():
    """Import the ingest script by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "ingest_episodes", ROOT / "scripts" / "ingest_episodes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSettings:
    def __init__(self, index: str, namespace: str) -> None:
        self.pinecone_index = index
        self.pinecone_namespace = namespace


def test_refuses_the_live_bullpen_index():
    ingest = _load_ingest()
    with pytest.raises(SystemExit) as exc:
        ingest.check_target(FakeSettings("bullpen-concierge", "mcg"))
    assert "bullpen-concierge" in str(exc.value)


@pytest.mark.parametrize(
    "namespace", ["podcast", "clawpump", "questions", "summaries", "assets"]
)
def test_refuses_every_namespace_the_bullpen_project_uses(namespace):
    ingest = _load_ingest()
    with pytest.raises(SystemExit):
        ingest.check_target(FakeSettings("mcg-search", namespace))


def test_allows_the_mcg_target():
    ingest = _load_ingest()
    ingest.check_target(FakeSettings("mcg-search", "mcg"))  # must not raise


def test_first_window_id_is_deterministic():
    """Re-indexing an episode must overwrite its rows, not duplicate them.

    That property is what makes the ingest safe to interrupt, and it
    holds only while the id is a pure function of (episode, start).
    """
    from app.schemas import Episode

    ingest = _load_ingest()
    ep = Episode(episode_id="abc123", title="t",
                 url="https://www.youtube.com/watch?v=abc123",
                 segments=[{"t": 4.5, "text": "hello"}])
    assert ingest.first_window_id(ep) == ingest.first_window_id(ep)
    assert len(ingest.first_window_id(ep)) == 32


def test_first_window_id_differs_per_episode():
    from app.schemas import Episode

    ingest = _load_ingest()

    def ep(eid):
        return Episode(episode_id=eid, title="t", url="https://x/watch?v=1",
                       segments=[{"t": 0.0, "text": "hello"}])

    assert ingest.first_window_id(ep("a")) != ingest.first_window_id(ep("b"))


def test_first_window_id_survives_an_episode_with_no_segments():
    """Must not raise. An empty episode is odd, not fatal."""
    from app.schemas import Episode

    ingest = _load_ingest()
    ep = Episode(episode_id="empty", title="t", url="https://x/watch?v=1",
                 segments=[])
    assert len(ingest.first_window_id(ep)) == 32
