"""The MCG search API and the page it serves.

Endpoints:

    GET  /                       the page
    GET  /v1/episodes            what is indexed, newest first
    POST /v1/search              answer in one response (bots, scripts)
    POST /v1/search/stream       the same answer as SSE (the page)
    GET  /v1/health              liveness, for a platform health check

The page uses the streaming route so citations can be painted as soon as
retrieval finishes, about a second in, rather than after the answer. The
answer itself arrives in one piece by default: through the proxy a
streamed call takes ~19s to reach its first token where the whole
non-streamed call returns in ~4.4s. See stream_answers in app/config.py.
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .episode_store import load as load_episodes
from .search import (
    MARKET_CALL_PREFIX,
    REFUSAL_ANSWER,
    MCGIndex,
    _already_declines,
    check_credentials,
    is_market_call,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
EPISODES = ROOT / "data" / "episodes.json"
EPISODE_INDEX = ROOT / "data" / "episode_index.json"
SUMMARIES = ROOT / "data" / "summaries.json"

app = FastAPI(title="MCG Search", docs_url=None, redoc_url=None)

# One index for the process. Building it opens clients and loads settings;
# doing that per request would add latency to a path that already has too
# much of it.
_index: MCGIndex | None = None


def index() -> MCGIndex:
    global _index
    if _index is None:
        _index = MCGIndex()
    return _index


class SearchBody(BaseModel):
    # Bounded because input is billed per token and the request body is
    # the one part of the cost a stranger controls.
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/v1/health")
async def health() -> dict:
    return {"ok": True, "model": get_settings().search_model}


@app.get("/v1/episodes")
async def episodes() -> dict:
    """Titles and dates only, never the transcripts.

    Prefers the committed 85KB index so a deploy works from a clean
    checkout — data/episodes.json is 19MB and deliberately gitignored,
    since it is re-fetchable and the searchable copy is in Pinecone
    anyway. Falls back to building the list from the full file on a
    laptop that has one but has not run build_index_json yet.
    """
    if EPISODE_INDEX.exists():
        rows = json.loads(EPISODE_INDEX.read_text())
    elif EPISODES.exists():
        from scripts.build_index_json import build
        rows = build(load_episodes(EPISODES))
    else:
        return {"count": 0, "hours": 0, "episodes": []}
    total = sum(r["seconds"] for r in rows)
    return {"count": len(rows), "hours": round(total / 3600), "episodes": rows}


@app.get("/v1/summaries")
async def summaries() -> dict:
    """A TL;DR and timestamped topics per episode.

    Search answers a question somebody already thought to ask. This is
    the other half: how they find the January interview about the thing
    they are researching today, in an archive where the median episode
    has about sixty views.

    Served whole — 820KB for 404 episodes — because the page filters
    client-side and a round trip per expanded card would be slower than
    sending it once.
    """
    if not SUMMARIES.exists():
        return {"count": 0, "summaries": {}}
    data = json.loads(SUMMARIES.read_text())
    return {"count": len(data), "summaries": data}


@app.post("/v1/search")
async def search(body: SearchBody) -> dict:
    try:
        result = await index().search(body.query, body.top_k)
    except Exception as exc:                                  # noqa: BLE001
        logger.exception("search failed")
        raise HTTPException(502, "the model call failed") from exc
    return json.loads(result.model_dump_json())


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/search/stream")
async def search_stream(body: SearchBody) -> StreamingResponse:
    """Citations first, then the answer token by token.

    Retrieval takes about a second and the answer takes tens of seconds,
    so the hits are sent as soon as they exist. The reader gets something
    real -- the episodes and moments being read -- while the answer is
    still being written.
    """
    async def stream():
        query = body.query.strip()
        idx = index()

        from .search import EMPTY_QUERY_ANSWER, MIN_QUERY_CHARS
        if len(query) < MIN_QUERY_CHARS:
            yield _sse("hits", {"hits": []})
            yield _sse("delta", {"text": EMPTY_QUERY_ANSWER})
            yield _sse("done", {"refused": False})
            return

        try:
            hits = await idx.retrieve(query, body.top_k)
        except Exception:                                     # noqa: BLE001
            logger.exception("retrieval failed")
            yield _sse("error", {"message": "search is unavailable right now"})
            return

        yield _sse("hits", {"hits": [json.loads(h.model_dump_json())
                                     for h in hits]})

        refused = False
        try:
            if get_settings().stream_answers:
                async for chunk in idx.answer_stream(query, hits):
                    if chunk == "\x00REFUSAL\x00":
                        refused = True
                        continue
                    yield _sse("delta", {"text": chunk})
            else:
                # One call, one delta. The SSE contract is unchanged, so
                # the page does not care which of these produced the text
                # — but through the proxy this reaches a COMPLETE answer
                # in about the time the streamed one takes to emit its
                # first token. See stream_answers in app/config.py.
                response = await idx._answer(query, hits)
                if response.stop_reason == "refusal":
                    refused = True
                    yield _sse("delta", {"text": REFUSAL_ANSWER})
                else:
                    text = "".join(b.text for b in response.content
                                   if b.type == "text")
                    # The page is the surface where a screenshot happens,
                    # so it gets the same guard the API answer does.
                    if is_market_call(query) and not _already_declines(text):
                        text = f"{MARKET_CALL_PREFIX}\n\n{text}"
                    yield _sse("delta", {"text": text})
        except Exception:                                     # noqa: BLE001
            # Bytes may already be on the wire, so a retry would duplicate
            # the answer. End honestly instead of pretending to finish.
            logger.exception("answer stream failed")
            yield _sse("error", {"message": "the answer stopped early"})
            return

        yield _sse("done", {"refused": refused})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx and friends buffer SSE by default, which turns a
            # streamed answer back into a 40-second blank page.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(WEB / "favicon.svg", media_type="image/svg+xml")


async def _warm() -> None:
    """Check the config can work, then build the client.

    The credential check is deliberately NOT caught: a server that
    cannot reach a model should refuse to start rather than accept
    traffic and fail every request with an SDK error nobody can read.
    """
    check_credentials(get_settings())
    try:
        await asyncio.to_thread(index)
    except Exception:                                         # noqa: BLE001
        logger.exception("warm-up failed; will retry on first request")


@app.on_event("startup")
async def startup() -> None:
    await _warm()
