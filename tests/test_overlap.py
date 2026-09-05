"""Deciding what a live stream adds that the clips do not already have.

The 31 Aug stream contained the interviews published on 1 Sep and none of
the nine published the week before. So the question for every stream
window is not "is this a stream" but "has this exact conversation already
been indexed as a clip" — and getting that wrong in either direction
costs something real: keeping duplicates makes retrieval return the same
words twice, dropping too much loses the only copy of the market talk
that is not published anywhere else.
"""

from app.overlap import (
    DUPLICATE_AT,
    duplicate_fraction,
    is_duplicate,
    nearby,
    reference_shingles,
    shingles,
)

CLIP = ("so the thing about this protocol is that it settles onchain and "
        "the fees go back to the people who actually provide liquidity "
        "which is the part nobody else has managed to get right yet")


def ep(text, published_at="2026-09-01"):
    return {"published_at": published_at,
            "segments": [{"t": 0.0, "text": text}]}


def test_the_same_speech_is_recognised():
    ref = reference_shingles([ep(CLIP)])
    assert is_duplicate(CLIP, ref)


def test_a_retranscription_still_matches():
    """The two captions are not identical — YouTube re-runs auto-caption
    per video, so punctuation and the odd word differ. Runs of ordinary
    words are what survive that, which is why the comparison is on
    shingles rather than on strings."""
    ref = reference_shingles([ep(CLIP)])
    retranscribed = CLIP.replace("onchain", "on chain").replace("yet", "yet.")
    assert duplicate_fraction(retranscribed, ref) > DUPLICATE_AT


def test_unrelated_market_talk_is_kept():
    """The reason to index streams at all: hours of commentary that
    exists in no clip. Dropping it would make the whole exercise
    pointless."""
    ref = reference_shingles([ep(CLIP)])
    market = ("bitcoin is holding this level and the funding has flipped "
              "negative which usually means the shorts are about to get "
              "squeezed out of their positions before any real move")
    assert not is_duplicate(market, ref)


def test_a_boundary_window_counts_as_duplicate():
    """Half a window of clip and half of stream: the clip already has
    the shared part, so dropping it loses nothing."""
    ref = reference_shingles([ep(CLIP)])
    half = CLIP + " and now back to the charts for a moment"
    assert duplicate_fraction(half, ref) >= DUPLICATE_AT


def test_nothing_to_compare_against_keeps_the_window():
    """An unmatched window is kept. Storing something twice is a smaller
    mistake than silently dropping its only copy."""
    assert duplicate_fraction(CLIP, set()) == 0.0
    assert not is_duplicate(CLIP, set())


def test_short_text_is_never_called_a_duplicate():
    ref = reference_shingles([ep(CLIP)])
    assert duplicate_fraction("yeah exactly", ref) == 0.0


def test_only_episodes_near_the_stream_are_compared():
    """The clips go up a day or two later. Comparing against the whole
    archive would mean holding millions of shingles for no gain."""
    eps = [ep("a", "2026-09-01"), ep("b", "2026-08-30"),
           ep("c", "2026-06-01"), ep("d", None)]
    got = nearby(eps, "2026-08-31")
    assert len(got) == 2                       # the two within a week


def test_an_undated_stream_compares_against_nothing():
    """Better to index a duplicate than to guess which clips it holds."""
    assert nearby([ep("a", "2026-09-01")], None) == []


def test_shingles_are_long_enough_to_mean_something():
    """Eight words of spontaneous speech repeating by chance is
    vanishingly unlikely; three would happen constantly."""
    a = shingles("the market is up today and everyone is very happy")
    b = shingles("the market is down today and everyone is very sad")
    assert not (a & b)


# --- summaries: a filler word is not the founder's name ----------------

def test_being_addressed_is_not_a_name():
    """One summary opened "Dude, founder of RatSpeak" because a caption
    had somebody say it. A TL;DR that misnames the founder in its first
    word is worse than one saying "the founder" — it reads as
    confidently wrong."""
    from app.summaries import looks_misattributed
    for bad in ["Dude, founder of RatSpeak, discusses...",
                "Bro explains the fee split",
                "Guys from the team walk through it",
                "yeah so the founder says"]:
        assert looks_misattributed(bad), bad


def test_a_real_name_or_a_role_is_left_alone():
    from app.summaries import looks_misattributed
    for good in ["Dakota, the founder of Sinjoh, explains...",
                 "The founder of Ratspeak discusses...",
                 "Ratspeak is an offline-capable mesh network",
                 "Duncan walks through the fee split"]:
        assert not looks_misattributed(good), good
