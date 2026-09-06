"""Every link on the page points at something real.

The worked example on the homepage shipped with a video id I had simply
made up — lTOZ5b3Zt2E, which is in no episode of the archive. It looked
right, the player opened, and it went to a video that has nothing to do
with the quote above it.

That is precisely the failure the system prompt spends four sentences
forbidding the model from committing ("a fabricated link in a citation
is worse than no link: it looks checkable and is not"), and it was
sitting in hand-written HTML where no rule applied.

So the page gets checked the same way the answers do.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "web" / "index.html").read_text()
INDEX = json.loads((ROOT / "data" / "episode_index.json").read_text())
BY_ID = {e["id"]: e for e in INDEX}


def hardcoded_citations():
    """(video id, seconds) for every timestamp button written into HTML."""
    return re.findall(r'data-vid="([\w-]+)"\s+data-t="(\d+)"', PAGE)


def test_the_page_has_a_worked_example():
    """Shown before anyone types. Losing it silently would remove the
    one thing that demonstrates the product at zero effort."""
    assert hardcoded_citations()


@pytest.mark.parametrize("vid,secs", hardcoded_citations())
def test_every_hardcoded_video_id_is_a_real_episode(vid, secs):
    assert vid in BY_ID, (
        f"{vid} is in no episode of the archive — a fabricated link on "
        f"the homepage is the exact thing the prompt forbids the model "
        f"from doing"
    )


@pytest.mark.parametrize("vid,secs", hardcoded_citations())
def test_every_hardcoded_timestamp_is_inside_its_episode(vid, secs):
    """A real id and a time past the end is still a broken citation:
    YouTube opens the video and ignores the seek."""
    runtime = BY_ID[vid]["seconds"]
    assert int(secs) < runtime, (
        f"{vid} is {runtime}s long but the page seeks to {secs}s"
    )


def test_the_example_quotes_the_timestamps_it_links():
    """The prose cites 26:10 and 9:56; the buttons must be those, or the
    reader presses one and lands somewhere the quote is not."""
    shown = set(re.findall(r"<b>(\d{1,2}:\d{2})</b>", PAGE))
    linked = {f"{int(s) // 60}:{int(s) % 60:02d}" for _v, s in hardcoded_citations()}
    assert shown == linked, f"prose cites {shown}, buttons link {linked}"


def test_no_raw_youtube_links_in_the_prose():
    """The page links through the player, never as bare anchors — the
    whole point is that pressing a moment plays it here."""
    body = PAGE.split("<footer")[0]
    assert "youtube.com/watch" not in body


def test_the_disclaimer_survives():
    """MCG's own line from their channel description, and the most
    important sentence on a page built over 406 founders pitching their
    own tokens."""
    assert "endorsement" in PAGE.lower()
    assert "not financial advice" in PAGE.lower()


def test_transcript_text_is_never_written_as_html():
    """Titles and transcripts are third-party caption text. They reach
    the DOM through textContent so a crafted title cannot become markup."""
    assert ".textContent = h.title" in PAGE
    assert ".textContent = h.text" in PAGE


def test_only_one_player_can_be_open():
    """Found by ear: clicking two timestamps left two videos playing, so
    two founders talked over each other. Hiding the element is not
    enough — the iframe has to be removed to stop the audio."""
    assert "closeAllPlayers" in PAGE
    assert "closeAllPlayers(player)" in PAGE       # opening closes others
    assert PAGE.count('r.innerHTML = ""') >= 1     # removes, not just hides


def test_a_new_search_stops_playback():
    """Asking a new question while a clip plays should silence it."""
    submit = PAGE.split('#f").onsubmit')[1]
    assert "closeAllPlayers()" in submit.split("fetch(")[0]


# --- summaries on the page ---------------------------------------------

def test_the_page_loads_summaries():
    """404 summaries generated and not shown would be the whole feature
    sitting in a file nobody reads."""
    assert "/v1/summaries" in PAGE
    assert "SUMMARIES" in PAGE


def test_a_missing_summaries_endpoint_does_not_break_the_list():
    """Without summaries the list should still render, just not expand.
    A page that fails to load because an optional extra is missing is
    worse than a plain page."""
    assert "catch(() => ({ summaries: {} }))" in PAGE


def test_every_topic_is_a_play_button():
    """The reason topics are stored as {t, text} rather than as prose
    with times written into it."""
    assert "start=${tp.t}" in PAGE


def test_collapsing_a_card_stops_the_audio():
    """Hiding the element leaves the iframe playing, which is how three
    founders ended up talking over each other."""
    seg = PAGE.split("const open = panel.classList.toggle")[1][:400]
    assert 'innerHTML = ""' in seg


# --- counts, which is the other way the page can be confidently wrong ---

def test_the_headline_is_not_a_typed_in_number():
    """The hero read "Ask 406 interviews anything" over an archive of
    458 videos, for as long as the streams had been indexed. Nothing on
    the page contradicted it, because nothing on the page knew.

    A number about the archive belongs to the archive. This checks the
    visible copy carries none of its own — the counts arrive from
    /v1/episodes, the same source the episode list already trusted."""
    body = PAGE.split("<body>", 1)[1]
    # Strip comments and script, where a number is either explanation or
    # is being read from the data rather than asserted to the reader.
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    visible = re.sub(r"<[^>]+>", " ", body)
    # Three or more digits is a claim about scale. Times (12:30) and
    # small numbers are not, and neither is a year in a dated example.
    claims = [n for n in re.findall(r"\b\d{3,}\b", visible)
              if not re.fullmatch(r"(19|20)\d\d", n)]
    assert not claims, f"hardcoded counts in visible copy: {claims}"


def test_the_page_names_both_kinds_of_video():
    """458 is only meaningful split: 41-minute founder interviews and
    3-hour broadcasts are different things to search."""
    assert "by_format" in PAGE
    assert "founder interviews" in PAGE
    assert "live streams" in PAGE


def test_the_status_line_survives_a_slow_episodes_call():
    """ARCHIVE_COUNT is 0 until /v1/episodes answers, and somebody can
    search before then. Printing "searching 0 episodes" would be a lie
    told by a loading state."""
    assert 'ARCHIVE_COUNT ? `${ARCHIVE_COUNT} episodes` : "the archive"' in PAGE
