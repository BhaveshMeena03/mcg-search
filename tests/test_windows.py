"""Windowing, and the invariant the citations depend on.

The embedded text and the timestamped text must contain the same words.
If a future change leaks timestamps into the embedded copy, retrieval
quality shifts and nothing fails — the answers just get slightly worse
for a reason nobody can see. That is the regression worth a test.
"""

import re

from app.schemas import TranscriptSegment
from app.search import _deep_link, _timestamp, _windows, stamp


def segs(*pairs) -> list[TranscriptSegment]:
    return [TranscriptSegment(t=t, text=text) for t, text in pairs]


def test_embedded_text_carries_no_timestamps():
    """The invariant. Timestamps live in the rebuilt copy only."""
    windows = _windows(segs((0.0, "alpha"), (5.0, "beta")), 2400, 2)
    for _start, text, times in windows:
        assert not re.search(r"\[\d+:\d\d\]", text)
        assert re.search(r"\[\d+:\d\d\]", stamp(text, times))


def test_the_rebuilt_copy_holds_exactly_the_same_words():
    """Stronger than the old assertion, and the reason storing the
    second copy was waste: it is recoverable from the text and the
    times, so ~2400 bytes of duplicate travelled on every one of the 50
    rerank candidates a query pulls."""
    windows = _windows(segs((0.0, "alpha"), (5.0, "beta"), (9.0, "gamma")),
                       2400, 2)
    for _start, text, times in windows:
        stripped = re.sub(r"\[\d+:\d{2}(?::\d{2})?\] ", "",
                          stamp(text, times))
        assert stripped == text


def test_stamp_falls_back_when_the_times_are_missing_or_wrong():
    """Vectors written before this change carry no line_times, and a
    malformed row must degrade to a coarser citation rather than an
    exception on the request path."""
    assert stamp("a\nb", None) == "a\nb"
    assert stamp("a\nb", []) == "a\nb"
    assert stamp("a\nb", [0.0]) == "a\nb"          # length mismatch
    assert stamp("a\nb", [0.0, 5.0, 9.0]) == "a\nb"


def test_stamp_is_much_smaller_than_the_copy_it_replaces():
    windows = _windows(segs(*[(float(i * 5), "x" * 90) for i in range(20)]),
                       2400, 2)
    _start, text, times = windows[0]
    # A float per line against a full second copy of the passage.
    assert len(str(times)) < len(stamp(text, times)) / 4


def test_window_starts_at_its_first_segment():
    windows = _windows(segs((12.0, "alpha"), (14.0, "beta")), 2400, 2)
    assert windows[0][0] == 12.0


def test_respects_the_char_budget():
    many = segs(*[(float(i), "x" * 100) for i in range(20)])
    for _start, text, _times in _windows(many, 300, 0):
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


def test_times_pack_into_something_pinecone_accepts():
    """Pinecone metadata takes a string, number, boolean or list of
    strings. A list of floats is a 400, which is how this was found."""
    from app.search import pack_times
    assert pack_times([0.0, 7.72, 14.16]) == "0,7,14"
    assert isinstance(pack_times([1.5]), str)
    assert pack_times([]) == ""


def test_stamp_reads_the_packed_string_back():
    from app.search import pack_times
    times = [0.0, 7.72, 14.16]
    assert stamp("a\nb\nc", pack_times(times)) == "[0:00] a\n[0:07] b\n[0:14] c"


def test_stamp_survives_a_corrupt_packed_string():
    assert stamp("a\nb", "not,numbers") == "a\nb"
    assert stamp("a\nb", "0") == "a\nb"          # wrong length
