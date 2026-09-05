"""The shapes that move through the pipeline.

Deliberately small: MCG is search only, so this needs the episode side
and nothing else.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """One timestamped line, as it comes out of a caption file."""

    t: float = Field(..., ge=0)   # start time in seconds
    text: str


class Episode(BaseModel):
    """One MCG episode with a timestamped transcript.

    On this channel that is one project interview, which is the reason
    indexing it is worth doing at all — a question about a project has its
    answer inside one episode rather than scattered across forty.
    """

    episode_id: str
    title: str
    url: str
    platform: Literal["youtube"] = "youtube"
    published_at: str | None = None
    segments: list[TranscriptSegment]


class Hit(BaseModel):
    """A retrieved transcript window, deep-linked to the moment."""

    episode_id: str
    title: str
    start_seconds: float
    timestamp: str                 # "14:32"
    deep_link: str                 # url that jumps to start_seconds
    text: str
    # The same passage with a timestamp on every line. Only the model reads
    # this — showing a reader the same words twice would double the payload
    # for nothing. Excluded from API responses.
    text_ts: str = Field(default="", exclude=True)
    published_at: str | None = None
    score: float


class SearchResponse(BaseModel):
    answer: str
    hits: list[Hit]
    model: str
    refused: bool = False
