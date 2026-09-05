"""Concurrent writes to data/episodes.json.

This is the file a 3.5-hour fetch writes into one episode at a time. The
bug this guards against has already happened once in practice: a
long-running writer merged against the snapshot it started with and
silently reverted everything finished in between.
"""

import json

import pytest

from app.episode_store import load, merge


def ep(eid: str, title: str = "t", published: str | None = None) -> dict:
    return {"episode_id": eid, "title": title, "url": f"https://x/watch?v={eid}",
            "platform": "youtube", "published_at": published,
            "segments": [{"t": 0.0, "text": "hello"}]}


def test_merge_adds(tmp_path):
    path = tmp_path / "episodes.json"
    assert merge([ep("a")], path) == (1, 0)
    assert [e["episode_id"] for e in load(path)] == ["a"]


def test_merge_keeps_what_is_already_there(tmp_path):
    path = tmp_path / "episodes.json"
    merge([ep("a")], path)
    merge([ep("b")], path)
    assert {e["episode_id"] for e in load(path)} == {"a", "b"}


def test_merge_rereads_and_does_not_revert_concurrent_work(tmp_path):
    """The actual bug. A writer holding a stale snapshot must not undo
    an episode another writer added after that snapshot was taken."""
    path = tmp_path / "episodes.json"
    merge([ep("a")], path)
    stale = load(path)                 # snapshot taken "hours ago"
    merge([ep("b")], path)             # another process finishes first
    merge(stale + [ep("c")], path)     # slow writer finally lands
    assert {e["episode_id"] for e in load(path)} == {"a", "b", "c"}


def test_merge_updates_in_place(tmp_path):
    path = tmp_path / "episodes.json"
    merge([ep("a", title="old")], path)
    assert merge([ep("a", title="new")], path) == (0, 1)
    assert load(path)[0]["title"] == "new"
    assert len(load(path)) == 1


def test_sorted_newest_first(tmp_path):
    path = tmp_path / "episodes.json"
    merge([ep("old", published="2026-01-01"),
           ep("new", published="2026-09-01")], path)
    assert [e["episode_id"] for e in load(path)] == ["new", "old"]


def test_load_of_a_missing_file_is_empty_not_an_error(tmp_path):
    assert load(tmp_path / "nope.json") == []


def test_refuses_to_overwrite_unreadable_json(tmp_path):
    """Never silently start from nothing.

    An unreadable file that gets replaced with one episode is how a
    catalogue of 407 disappears.
    """
    path = tmp_path / "episodes.json"
    path.write_text("{ this is not json")
    with pytest.raises(RuntimeError):
        load(path)


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "episodes.json"
    merge([ep("a")], path)
    assert json.loads(path.read_text())
    assert not (tmp_path / "episodes.json.tmp").exists()
