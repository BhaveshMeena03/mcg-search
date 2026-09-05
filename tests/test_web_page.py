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
