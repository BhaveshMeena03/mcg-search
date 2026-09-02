"""Windowing, and the invariant the citations depend on.

The embedded text and the timestamped text must contain the same words.
If a future change leaks timestamps into the embedded copy, retrieval
quality shifts and nothing fails — the answers just get slightly worse
for a reason nobody can see. That is the regression worth a test.
"""

import re

from app.schemas import TranscriptSegment
from app.search import _deep_link, _timestamp, _windows


def segs(*pairs) -> list[TranscriptSegment]:
    return [TranscriptSegment(t=t, text=text) for t, text in pairs]


def test_embedded_text_carries_no_timestamps():
    """The invariant. Timestamps belong in `stamped` only."""
    windows = _windows(segs((0.0, "alpha"), (5.0, "beta")), 2400, 2)
    for _start, text, stamped in windows:
        assert not re.search(r"\[\d+:\d\d\]", text)
        assert re.search(r"\[\d+:\d\d\]", stamped)


def test_the_two_copies_hold_the_same_words():
    windows = _windows(segs((0.0, "alpha"), (5.0, "beta"), (9.0, "gamma")),
                       2400, 2)
    for _start, text, stamped in windows:
        stripped = re.sub(r"\[\d+:\d{2}(?::\d{2})?\] ", "", stamped)
        assert stripped == text


def test_window_starts_at_its_first_segment():
    windows = _windows(segs((12.0, "alpha"), (14.0, "beta")), 2400, 2)
    assert windows[0][0] == 12.0


def test_respects_the_char_budget():
    many = segs(*[(float(i), "x" * 100) for i in range(20)])
    for _start, text, _stamped in _windows(many, 300, 0):
        assert len(text) <= 300 + 100  # one segment may overshoot; see below


def test_segment_longer_than_the_budget_is_kept_not_dropped():
    """A single over-long segment must still be indexed.

    Dropping it would make a stretch of the episode unreachable, and it
    would happen silently — the ingest count would just be lower.
    """
    windows = _windows(segs((0.0, "y" * 5000)), 2400, 2)
    assert len(windows) == 1
    assert len(windows[0][1]) == 2400


def test_windows_overlap_so_a_straddling_answer_survives():
    many = segs(*[(float(i), f"seg{i}") for i in range(12)])
    windows = _windows(many, 20, overlap_segments=2)
    assert len(windows) > 1
    # Consecutive windows must share text, or an answer spanning a
    # boundary is retrievable from neither side.
    assert any(
        set(a[1].split()) & set(b[1].split())
        for a, b in zip(windows, windows[1:], strict=False)
    )


def test_windowing_terminates_on_pathological_input():
    """The overlap step must always advance.

    i = max(j - overlap, i + 1) is what guarantees it. A regression here
    is an infinite loop during ingest, not a wrong answer.
    """
    many = segs(*[(float(i), "z" * 50) for i in range(40)])
    assert len(_windows(many, 10, overlap_segments=100)) == 40


def test_no_segments():
    assert _windows([], 2400, 2) == []


def test_timestamp_formatting():
    assert _timestamp(0) == "0:00"
    assert _timestamp(65) == "1:05"
    assert _timestamp(3600) == "1:00:00"
    assert _timestamp(3661) == "1:01:01"
    assert _timestamp(59.9) == "0:59"      # truncates, never rounds forward


def test_deep_link_lands_on_the_second():
    url = "https://www.youtube.com/watch?v=abc123"
    assert _deep_link(url, 92.7) == f"{url}&t=92s"


def test_deep_link_joins_correctly_without_a_query_string():
    assert _deep_link("https://youtu.be/abc", 30) == "https://youtu.be/abc?t=30s"
