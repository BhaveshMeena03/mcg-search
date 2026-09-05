"""Condense an episode into a TL;DR and a list of timestamped topics.

406 episodes averaging around sixty views is a lot of archive earning
nothing. Search answers a question somebody already thought to ask; a
summary is how they find the January interview about the thing they are
researching today. It is the browse half of the same problem.

Two decisions worth knowing.

TOPICS ARE STRUCTURED, not prose with times written into it. Each is
{t, text}, so the page can make every line a play button rather than
printing a timestamp the reader has to go and find. It also means a
topic can be checked: a time outside the episode is a bug we can catch,
where a number inside a paragraph is just text.

THE MODEL IS GIVEN A STAMPED TRANSCRIPT and asked to work only from it.
Same discipline as the answers: an invented timestamp that looks
plausible is worse than a missing one, because the reader presses it and
lands somewhere the topic is not. validate() throws those away rather
than shipping them.
"""

import json
import logging
import re
from dataclasses import dataclass

from .schemas import Episode
from .search import _timestamp

logger = logging.getLogger(__name__)

# Enough for a 40-minute interview at roughly 150 words a minute. The
# longest episodes in the archive run past two hours; those get clipped,
# which costs the tail of a long conversation rather than the summary.
MAX_TRANSCRIPT_CHARS = 120_000

SUMMARY_PROMPT = """\
You are summarising one episode of the MCG podcast, which interviews one \
crypto project per episode. You are given the transcript with a timestamp \
in front of every line.

Return ONLY a JSON object, no prose around it, shaped exactly like this:

{"tldr": "...", "topics": [{"t": 125, "text": "..."}, ...]}

tldr: two or three sentences. What the project does, who is talking, and \
the single most interesting or surprising thing in the conversation. \
Write it for somebody deciding whether to spend forty minutes here, not \
as a description of a video. Name the project. Do not open with "In this \
episode" or "This episode covers".

topics: between five and ten entries, in the order they occur.
  t     seconds from the start, an integer. Take it from the timestamp \
in front of the line the topic begins at. NEVER estimate or round to a \
neat number: a reader presses this and expects to land on it. If you are \
not certain which line a topic starts at, leave the topic out.
  text  one sentence, specific. "The fee split: 1% of volume, 0.35% to \
the platform" not "They discuss fees". Numbers, names and claims are the \
whole value — a topic list of vague nouns is worth nothing.

Rules:
- Use ONLY the transcript. Do not add anything you know about the project \
from elsewhere.
- These are auto-generated captions with no speaker labels, and an \
interview has at least two people in it. Say "the founder" or "the host" \
unless the transcript itself makes a name unambiguous. Being ADDRESSED \
is not a name: "dude", "bro", "man", "boss", "guys" are how people talk \
to each other, and one summary opened "Dude, founder of RatSpeak" \
because the transcript had somebody say it. Unusual project \
names are often transcribed wrongly; use the spelling from the episode \
title, which is human-written.
- This is informational, never investment advice. Report what was \
claimed, attributed to whoever claimed it. Never endorse a project, \
assess a token, or repeat a projection as though it were a fact.
- If the transcript is too broken or too short to summarise honestly, \
return {"tldr": "", "topics": []} rather than inventing something."""


@dataclass
class Summary:
    episode_id: str
    tldr: str
    topics: list[dict]
    model: str

    def to_dict(self) -> dict:
        return {"tldr": self.tldr, "topics": self.topics, "model": self.model}


def stamped_transcript(episode: Episode,
                       max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """The transcript with [h:mm:ss] in front of every line.

    The model can only cite a timestamp it was shown, which is the point:
    it makes an invented one a deviation from the input rather than a
    plausible guess.
    """
    out: list[str] = []
    total = 0
    for seg in episode.segments:
        line = f"[{_timestamp(seg.t)}] {seg.text}"
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


# Words that are how people address each other, not names. A summary
# opened "Dude, founder of RatSpeak" because a caption had somebody say
# it, and a TL;DR that misnames the founder in its first word is worse
# than one that says "the founder" — it reads as confidently wrong.
_NOT_A_NAME = re.compile(
    r"^(dude|bro|man|guys|buddy|boss|sir|mate|yeah|okay|hey)\b",
    re.IGNORECASE)


def looks_misattributed(tldr: str) -> bool:
    return bool(_NOT_A_NAME.match((tldr or "").strip()))


_JSON = re.compile(r"\{.*\}", re.S)


def parse(raw: str) -> tuple[str, list[dict]]:
    """Pull the object out of a reply that may be wrapped in prose."""
    match = _JSON.search(raw or "")
    if not match:
        return "", []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", []
    tldr = data.get("tldr")
    topics = data.get("topics")
    return (tldr if isinstance(tldr, str) else "",
            topics if isinstance(topics, list) else [])


def validate(topics: list[dict], episode: Episode) -> list[dict]:
    """Keep the topics that point at a moment the episode actually has.

    A timestamp past the end sends the reader to a video that ignores the
    seek; one that is merely wrong sends them somewhere the topic is not,
    which is worse because it looks right. Both get dropped rather than
    shipped — a shorter honest list beats a complete plausible one.
    """
    if not episode.segments:
        return []
    runtime = episode.segments[-1].t
    clean: list[dict] = []
    seen: set[int] = set()
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        text = str(topic.get("text", "")).strip()
        try:
            t = int(float(topic.get("t")))
        except (TypeError, ValueError):
            continue
        if not text or t < 0 or t > runtime:
            continue
        if t in seen:                     # two topics, one moment
            continue
        seen.add(t)
        clean.append({"t": t, "text": text})
    return sorted(clean, key=lambda x: x["t"])
