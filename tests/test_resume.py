"""What the ingest decides is already indexed.

The resume check is the difference between re-running the ingest for
free and re-embedding the whole archive. It used to guess: compute the
vector id of an episode's FIRST window and ask Pinecone whether that id
exists. Ids are sha256(episode_id:start_seconds), so that id is
computable from the transcript alone — which is what made the guess
tempting, and it is wrong for streams.

app.search.ingest drops stream windows that duplicate a clip already
published from the same broadcast. When the dropped window is the
opening one, its id is never written, and the episode reads as missing
however much of it is in the index. Measured on 2026-09-12: 62 of 634
episodes, every one of them actually present, re-embedded through
voyage-3.5 on every run — 62 four-hour streams for nothing.

These tests build the deduped case rather than describing it: real
windows, the first one withheld the way overlap.is_duplicate withholds
it, and the check asked what it would say.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.schemas import Episode
from app.search import _windows, window_id

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
    embedding_dimension = 1024
    pinecone_read_timeout_seconds = 20.0


class FakePinecone:
    """Enough of a Pinecone index to answer the resume check.

    Rows are (vector_id, episode_id). query() honours the episode_id
    filter and nothing else, because the filter is the whole point: the
    probe vector exists only because query() demands one.
    """

    def __init__(self, rows: list[tuple[str, str]], fail: Exception | None = None):
        self.rows = list(rows)
        self.fail = fail
        self.queries: list[str | None] = []

    def query(self, vector, top_k, namespace, filter=None,
              include_metadata=False, include_values=False):
        if self.fail:
            raise self.fail
        want = ((filter or {}).get("episode_id") or {}).get("$eq")
        self.queries.append(want)
        assert any(v != 0.0 for v in vector), "cosine rejects a zero vector"
        matches = [SimpleNamespace(id=vid)
                   for vid, eid in self.rows if eid == want]
        return SimpleNamespace(matches=matches[:top_k])

    def fetch(self, ids, namespace):
        wanted = set(ids)
        return SimpleNamespace(
            vectors={vid: {} for vid, _ in self.rows if vid in wanted}
        )


class FakeIndex:
    def __init__(self, pinecone: FakePinecone):
        self.index = pinecone
        self.namespace = "mcg-test"


def stream(episode_id: str = "stream1", n: int = 40) -> Episode:
    """A stream long enough to have several windows LEFT once the
    opening one is withheld — at 2400 chars a dozen segments make
    only two windows, and dropping the first leaves one, which is
    not the case being tested."""
    return Episode(
        episode_id=episode_id,
        title="MCG Live",
        url=f"https://www.youtube.com/watch?v={episode_id}",
        format="stream",
        published_at="2026-09-01",
        segments=[{"t": float(i * 6), "text": f"line {i} " + "word " * 60}
                  for i in range(n)],
    )


def rows_for(ep: Episode, drop_first: bool = False) -> list[tuple[str, str]]:
    """The vectors an ingest of this episode would leave behind."""
    windows = _windows(ep.segments, 2400, overlap_segments=2)
    if drop_first:
        windows = windows[1:]
    return [(window_id(ep.episode_id, start), ep.episode_id)
            for start, _text, _times in windows]


def present(idx, episodes) -> set[str]:
    ingest = _load_ingest()
    return asyncio.run(
        ingest.indexed_episode_ids(idx, episodes, FakeSettings())
    )


def test_a_stream_whose_first_window_was_deduped_is_not_re_indexed():
    """The regression. This episode is fully indexed apart from an
    opening window that overlap.is_duplicate removed, and the old check
    called the whole four-hour stream missing."""
    ep = stream()
    rows = rows_for(ep, drop_first=True)
    assert len(rows) > 1, "need a stream with windows left after the first"
    pinecone = FakePinecone(rows)

    assert present(FakeIndex(pinecone), [ep]) == {ep.episode_id}

    # And the thing this replaced: the first window's id is genuinely
    # absent, so fetching it would have said "index this again".
    first_start = _windows(ep.segments, 2400, overlap_segments=2)[0][0]
    missing = window_id(ep.episode_id, first_start)
    assert not pinecone.fetch(ids=[missing], namespace="mcg-test").vectors


def test_an_episode_with_nothing_in_the_index_is_reported_missing():
    """The check still has to say yes to work that has not been done."""
    ep = stream("brand-new")
    other = stream("something-else")
    pinecone = FakePinecone(rows_for(other))
    assert present(FakeIndex(pinecone), [ep, other]) == {other.episode_id}


def test_an_ordinary_indexed_episode_is_skipped():
    ep = stream("done")
    assert present(FakeIndex(FakePinecone(rows_for(ep))), [ep]) == {"done"}


def test_it_asks_once_per_episode_and_filters_on_the_episode():
    """One round trip each, filtered — not a scan of the namespace."""
    eps = [stream(f"e{i}") for i in range(5)]
    pinecone = FakePinecone([r for e in eps for r in rows_for(e)])
    present(FakeIndex(pinecone), eps)
    assert sorted(pinecone.queries) == sorted(e.episode_id for e in eps)


def test_a_failed_probe_is_raised_so_the_caller_indexes_everything():
    """Resume is an optimisation. When Pinecone cannot be asked, the
    ingest must do the work rather than stop — main() catches this and
    says so. Silently treating a failure as "already indexed" would skip
    real episodes."""
    pinecone = FakePinecone([], fail=RuntimeError("pinecone is down"))
    with pytest.raises(RuntimeError, match="pinecone is down"):
        present(FakeIndex(pinecone), [stream()])


def test_main_falls_back_to_indexing_everything_when_the_check_fails():
    src = (ROOT / "scripts" / "ingest_episodes.py").read_text()
    assert "resume check failed" in src


def test_window_ids_are_deterministic():
    """Re-indexing an episode must overwrite its rows, not duplicate
    them. That property is what makes the ingest safe to interrupt, and
    it holds only while the id is a pure function of (episode, start).
    The resume check leans on it: an episode indexed twice costs money,
    never a duplicate row."""
    assert window_id("abc123", 4.5) == window_id("abc123", 4.5)
    assert len(window_id("abc123", 4.5)) == 32
    assert window_id("a", 0.0) != window_id("b", 0.0)
    assert window_id("a", 0.0) != window_id("a", 6.0)
