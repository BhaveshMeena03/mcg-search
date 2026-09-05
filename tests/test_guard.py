"""The guard that keeps a scheduled ingest off somebody else's data.

One Pinecone account can hold several unrelated projects, and a wrong
.env in a job nobody is watching is the only path from this codebase to
data that is not its own. The refusal is asserted rather than assumed.

Deliberately configuration rather than constants: this repo carries no
knowledge of what else shares the account, because a deployment's
neighbours are its operator's business.
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
    def __init__(self, index: str, namespace: str,
                 protected_indexes: str = "", protected_namespaces: str = ""):
        self.pinecone_index = index
        self.pinecone_namespace = namespace
        self.protected_indexes = protected_indexes
        self.protected_namespaces = protected_namespaces


def test_refuses_a_protected_index():
    ingest = _load_ingest()
    with pytest.raises(SystemExit) as exc:
        ingest.check_target(
            FakeSettings("other-project", "mcg", protected_indexes="other-project")
        )
    assert "other-project" in str(exc.value)


@pytest.mark.parametrize("namespace", ["alpha", "beta", "gamma"])
def test_refuses_a_protected_namespace(namespace):
    ingest = _load_ingest()
    with pytest.raises(SystemExit):
        ingest.check_target(FakeSettings(
            "mcg-search", namespace,
            protected_namespaces="alpha,beta,gamma"))


def test_allows_the_mcg_target():
    ingest = _load_ingest()
    ingest.check_target(FakeSettings("mcg-search", "mcg"))  # must not raise


def test_nothing_is_protected_by_default():
    """A fresh install must not trip over a guard it never configured."""
    ingest = _load_ingest()
    ingest.check_target(FakeSettings("anything", "at-all"))


def test_whitespace_and_empty_entries_are_ignored():
    ingest = _load_ingest()
    ingest.check_target(FakeSettings("mcg-search", "mcg",
                                     protected_indexes=" , ,"))
    with pytest.raises(SystemExit):
        ingest.check_target(FakeSettings("x", "mcg",
                                         protected_indexes=" x , y "))


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


# --- the scheduled sync is guarded like the manual ingest ---------------

def test_the_sync_script_checks_the_target_too():
    """A scheduled job writing to the wrong index is worse than a manual
    one, because nobody is watching when it happens."""
    src = (ROOT / "scripts" / "sync_new.py").read_text()
    assert "check_target(settings)" in src


def test_the_sync_script_writes_transcripts_before_indexing():
    """Fetching is the slow, rate-limited, blockable half. If indexing
    dies after it, the fetch must not have to happen again."""
    src = (ROOT / "scripts" / "sync_new.py").read_text()
    assert src.index("merge(fetched, DATA)") < src.index("idx.ingest")


def test_the_sync_script_refreshes_the_page_index():
    """data/episode_index.json is what a deploy serves. A sync that adds
    episodes to Pinecone but not to that file makes the page disagree
    with what is searchable."""
    src = (ROOT / "scripts" / "sync_new.py").read_text()
    assert "SLIM.write_text" in src
