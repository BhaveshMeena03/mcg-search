"""Caption parsing.

The overlap stripping is the part that would break silently: if it
regressed, transcripts would still parse, still ingest, and still answer
— just with every other sentence duplicated, which reads as a worse
model rather than as a bug.
"""

from app.captions import coalesce, parse_vtt

VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
so the thing about

00:00:03.000 --> 00:00:05.000
so the thing about this protocol is

00:00:05.000 --> 00:00:07.000
this protocol is that it settles onchain
"""


def test_strips_youtube_rolling_overlap():
    """Each cue repeats a prefix of the last. Only the new tail survives."""
    segments = parse_vtt(VTT)
    joined = " ".join(s["text"] for s in segments)
    assert joined == "so the thing about this protocol is that it settles onchain"


def test_keeps_timestamps_of_first_appearance():
    segments = parse_vtt(VTT)
    assert segments[0]["t"] == 1.0
    assert [s["t"] for s in segments] == sorted(s["t"] for s in segments)


def test_numeric_lines_survive():
    """A finance archive is full of years and prices.

    The cue body is taken as the lines AFTER the timestamp rather than by
    filtering out digits, so a spoken "2026" is kept.
    """
    vtt = "WEBVTT\n\n00:00:04.000 --> 00:00:06.000\n2026\n"
    assert parse_vtt(vtt) == [{"t": 4.0, "text": "2026"}]


def test_html_entities_and_bleeps():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
           "it&#39;s [ __ ] expensive\n")
    assert parse_vtt(vtt)[0]["text"] == "it's [expletive] expensive"


def test_coalesce_merges_fragments_under_the_gap():
    segments = [{"t": 0.0, "text": "a"}, {"t": 1.0, "text": "b"},
                {"t": 30.0, "text": "c"}]
    out = coalesce(segments, min_gap=6.0)
    assert out == [{"t": 0.0, "text": "a b"}, {"t": 30.0, "text": "c"}]


def test_coalesce_does_not_mutate_its_input():
    segments = [{"t": 0.0, "text": "a"}, {"t": 1.0, "text": "b"}]
    coalesce(segments)
    assert segments == [{"t": 0.0, "text": "a"}, {"t": 1.0, "text": "b"}]


def test_empty_input():
    assert parse_vtt("") == []
    assert coalesce([]) == []
