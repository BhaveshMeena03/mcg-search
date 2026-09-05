"""Find the parts of a live stream that were already published as a clip.

MCG streams for four hours — market commentary with founder interviews
inside it — and then uploads those interviews as standalone episodes a
day or two later. Measured on the 31 Aug stream against the interviews
around it:

    Umia Finance      7/8 sampled passages found inside the stream
    Dominion Market   5/8
    GWOOD             4/8
    the other nine    0/8

So indexing streams whole would store a large share of the archive
twice. That is worse than wasteful. Retrieval would return the same
conversation from two places, and the worse of the two would sometimes
win: a citation into a 42-minute episode about one project is a better
destination than the identical words two hours forty into a broadcast.

The comparison is on word shingles rather than exact strings because the
two transcriptions are not identical — YouTube re-runs auto-captioning
per video, so the same speech comes back with different punctuation and
the occasional different word. Runs of ordinary words survive that.
"""

from __future__ import annotations

import re

# Long enough that a shared run means shared speech rather than a common
# turn of phrase. Eight words of spontaneous talk repeating by chance is
# vanishingly unlikely; three would happen constantly.
SHINGLE = 8

# Above this fraction, a window is treated as already published. Chosen
# to be forgiving: a window that is half the same speech is the boundary
# of a clip, and dropping those loses nothing because the clip has them.
DUPLICATE_AT = 0.5

# How far either side of a stream to look for its clips. The clips go up
# a day or two later, so a week is generous; widening it costs only
# comparison time, but comparing against the whole archive would mean
# holding millions of shingles in memory for no gain.
NEARBY_DAYS = 7

_WORDS = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> list[str]:
    return _WORDS.sub(" ", (text or "").lower()).split()


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    """Every run of n consecutive words, as joined strings."""
    words = normalise(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _days_apart(a: str | None, b: str | None) -> int:
    """Whole days between two ISO dates, or a large number if unknown."""
    if not a or not b:
        return 10_000
    from datetime import date
    try:
        d1 = date.fromisoformat(a[:10])
        d2 = date.fromisoformat(b[:10])
    except ValueError:
        return 10_000
    return abs((d1 - d2).days)


def nearby(episodes: list[dict], published_at: str | None,
           days: int = NEARBY_DAYS) -> list[dict]:
    """Episodes published close enough to be clips of this stream."""
    return [e for e in episodes
            if _days_apart(e.get("published_at"), published_at) <= days]


def reference_shingles(episodes: list[dict]) -> set[str]:
    """One set covering every candidate clip, built once per stream."""
    out: set[str] = set()
    for ep in episodes:
        text = " ".join(s.get("text", "") for s in ep.get("segments", []))
        out |= shingles(text)
    return out


def duplicate_fraction(text: str, reference: set[str]) -> float:
    """How much of `text` already appears in the reference clips.

    Zero when there is nothing to compare — an unmatched window is kept,
    because the cost of storing something twice is smaller than the cost
    of silently dropping the only copy of something.
    """
    own = shingles(text)
    if not own or not reference:
        return 0.0
    return len(own & reference) / len(own)


def is_duplicate(text: str, reference: set[str],
                 threshold: float = DUPLICATE_AT) -> bool:
    return duplicate_fraction(text, reference) >= threshold
