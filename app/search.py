"""MCG episode search.

Ingests timestamped episode transcripts, embeds windows of consecutive
segments (keeping each window's start time), and answers questions with an
answer plus citations that deep-link to the exact second in the video.

Ported from the Market Bubble index. What came across unchanged is the
part that took the measuring: the windowing, the separate embedded vs
stamped text, and the prompt rules that were each written after a wrong
answer went out in public. What changed is anything specific to a
two-host broadcast, because MCG is one host interviewing one project.
"""

import asyncio
import hashlib
import logging
from urllib.parse import urlparse
from xml.sax.saxutils import escape, quoteattr

import voyageai
from anthropic import AsyncAnthropic
from pinecone import Pinecone

from .config import get_settings
from .embeddings import embed_query, embed_texts, rerank_order
from .schemas import Episode, Hit, SearchResponse, TranscriptSegment

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = ("I can't help with that one — try asking about "
                  "something discussed on the show.")

# What the model is told to say when the excerpts do not contain the
# answer. Callers need to recognise a miss, and the wording is the only
# signal: the retriever always returns its top_k, so a full hit list says
# nothing about whether any of it was relevant.
NOT_FOUND_ANSWER = "I couldn't find that in the episodes I've indexed"

SYSTEM_PROMPT = """\
You answer questions about the MCG podcast using ONLY the transcript \
excerpts provided in <excerpts> tags. Each MCG episode is one interview \
with one project, so an episode title usually names the project being \
discussed. Each excerpt is tagged with its episode, timestamp, and — when \
known — the date it aired. Excerpts are given oldest first.

Rules:
1. Answer strictly from the excerpts. If they don't contain the answer, say \
"I couldn't find that in the episodes I've indexed" — do not use outside \
knowledge and do not guess. You may know something about a project from \
elsewhere; that is not what this tool is for, and a plausible answer the \
archive does not support is the worst thing you can produce.
1a. If you cite a timestamp ANYWHERE in your answer, your FIRST sentence \
must be about what was said, not about what was not. This is mechanical, \
not a matter of taste. These openings are forbidden whenever a citation \
follows: "I couldn't find", "I don't see", "I didn't find", "There's no \
direct/specific statement", "Not in those words", "Nothing matching", \
"The excerpts don't contain". Reaching for one and then writing "However, \
around 1:46:25 he does discuss..." produces a correct answer wearing a \
denial, and the reader stops at the first sentence. Rule 1 is for when you \
cite NOTHING. If you cite something, open with it — "Around 27:09, X" — \
and put any shortfall at the END as a qualifier.
2. Cite the moment. Every line inside an excerpt begins with its own \
timestamp in square brackets, like [16:16]. Cite the timestamp of the line \
you actually used, NOT the `at` attribute on the excerpt — that is only \
where the passage begins, and it can be a minute or more before the moment \
you are describing. Mention the episode too. NEVER write a URL or a \
Markdown link of any kind. You are not given the video addresses and \
cannot know them, so writing one means inventing it. A fabricated link in \
a citation is worse than no link: it looks checkable and is not. Give the \
timestamp and the episode name in plain text and let the interface link it.
3. Mind the dates. If excerpts from different dates disagree, say so and \
give the order ("in May they were pre-launch; by July they had shipped") \
rather than blending them into one state of the project that was never \
true. When a question is about where something stands now, lean on the \
most recent excerpt and say how recent it is. Never present a stale \
roadmap as current — this archive is mostly early-stage projects, and \
they move.
4. Summarize faithfully. Do not put words in anyone's mouth or invent \
quotes — paraphrase what the excerpt actually says.
5. These are auto-generated captions with NO speaker labels. An MCG \
episode is an interview, so most excerpts contain two people: the host \
asking and someone from the project answering. You usually cannot tell \
which is which from the words alone. Attribute to a named person only \
when the excerpt itself makes it unambiguous — the name is said, or \
someone is addressed by it. Otherwise write "the host", "the founder", or \
"the team", using the project name from the episode title only when the \
excerpt establishes that the speaker is from that project. A misattributed \
quote is worse than a vague one: the person named did not say it, and the \
person who did gets no credit.
5a. A name in the QUESTION is not evidence about the excerpts. Asked "what \
did the Scopl founder say about fees", the excerpts do not become about \
that founder. The question tells you what someone wants to know, never who \
was speaking. If the excerpts do not establish it, say so plainly and \
answer about what they DO establish, even when that is less than the \
question asked for.
5b. Auto-captions mangle unusual names, and this archive is full of them. \
A project called Scopl may be transcribed as "Scopple" or "scope hall", \
and the caption is not evidence the name is different — it is evidence the \
transcriber guessed. Do not correct a project's name to something you \
recognise, and do not report a mangling as an alternative name or a \
rebrand. If a passage is clearly about the project asked for, answer from \
it; use the spelling from the episode title, which is human-written.
5c. A number belongs to the thing named on its OWN line. Excerpts come \
from different episodes and different projects sit beside each other, so \
carrying a figure across lines invents a claim nobody made. If a line \
gives a figure without naming what it is for, say that or leave it out; \
never supply the subject from a neighbouring line, from the episode title, \
or from the question. And a number someone says was REACHED is not a \
number they are projecting — keep the tense.
5d. When one speaker states a figure and another corrects it, the \
correction is the answer. Read a few lines PAST any number before \
reporting it. Hedges like "something like that", "I might be off", \
"roughly" mark a figure as unreliable: either use the corrected one or say \
the number was approximate. Never carry a unit — K, million, billion — \
from one thing onto another.
6. This is an informational search tool, not financial or investment \
advice. Many of these projects have tokens. Never add buy/sell \
recommendations or price predictions of your own, and never characterise a \
project as a good or bad investment.
7. Keep it tight and conversational — a couple of sentences plus the \
citation, not an essay."""


def _timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _deep_link(url: str, seconds: float) -> str:
    """A watch URL that jumps to the moment.

    YouTube only, because the whole channel is. The Market Bubble version
    branched on platform because half that show lives on X, where there is
    no timestamp parameter at all.
    """
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={int(seconds)}s"


def _windows(
    segments: list[TranscriptSegment], max_chars: int, overlap_segments: int
) -> list[tuple[float, str, str]]:
    """Pack consecutive segments into windows of <= max_chars.

    Returns (start_seconds, text, stamped) — the same passage twice.

    `text` is what gets embedded and what a reader sees. `stamped` is the
    same lines with each one prefixed by its own timestamp, and it exists
    only to be handed to the model when it writes an answer.

    They are separate on purpose, and this is the single most important
    detail in the file. A window is minutes of speech carrying one start
    time, so a model given only that can cite nothing else — on Market
    Bubble every citation landed up to a minute early, on a product whose
    promise is the exact second.

    Putting the timestamps into the embedded text would fix that and
    quietly change retrieval: roughly a tenth of each window becomes
    non-semantic tokens and every vector shifts. Keeping the embedded text
    identical means ranking is decided by speech alone.

    Windows overlap by a few segments so an answer that straddles a
    boundary is still retrievable.
    """
    windows: list[tuple[float, str, str]] = []
    i = 0
    n = len(segments)
    while i < n:
        start_t = segments[i].t
        parts: list[str] = []
        stamped_parts: list[str] = []
        length = 0
        j = i
        while j < n and length + len(segments[j].text) + 1 <= max_chars:
            parts.append(segments[j].text)
            stamped_parts.append(f"[{_timestamp(segments[j].t)}] {segments[j].text}")
            length += len(segments[j].text) + 1
            j += 1
        if j == i:  # single segment longer than max_chars — take it whole
            clipped = segments[i].text[:max_chars]
            parts.append(clipped)
            stamped_parts.append(f"[{_timestamp(segments[i].t)}] {clipped}")
            j = i + 1
        windows.append((start_t, "\n".join(parts), "\n".join(stamped_parts)))
        if j >= n:
            break
        i = max(j - overlap_segments, i + 1)
    return windows


PLACEHOLDER_KEY = "unused-auth-is-in-the-base-url"

# Hosts the real Anthropic credential may be sent to. Anything else is a
# third party, however trustworthy, and gets the placeholder instead.
_ANTHROPIC_HOSTS = ("api.anthropic.com",)


def _outbound_key(settings) -> str:
    """The api_key to send, given where the request is going.

    A base-URL swap is the documented way to use an inference proxy, and
    the obvious implementation forwards whatever key is configured. That
    sends a live sk-ant-... to another company's servers on every single
    request. Proxies that authenticate on a path token — usepod does —
    ignore the key entirely, so there is nothing to lose by withholding
    it and a real credential to lose by not.

    Compare the parsed HOST, never a substring of the URL. The first
    version of this asked whether "api.anthropic.com" appeared anywhere
    in the base URL, which hands the key to
    https://api.anthropic.com.evil.example/v1 — a domain anyone can
    register. A test caught it; the substring form is the bug.
    """
    base = (settings.anthropic_base_url or "").strip()
    if not base:
        return settings.anthropic_api_key
    host = (urlparse(base).hostname or "").lower()
    if host in _ANTHROPIC_HOSTS:
        return settings.anthropic_api_key
    return PLACEHOLDER_KEY


class MCGIndex:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._voyage = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        self._anthropic = AsyncAnthropic(
            # Never hand the real Anthropic key to a third-party endpoint.
            #
            # Routing through a proxy is a base-URL swap, and the naive
            # version of that quietly ships sk-ant-... to whoever owns the
            # new host. usepod authenticates on a token in its URL path
            # and documents the api_key as ignored, so forwarding it buys
            # nothing and risks a live credential. Anything that is not
            # Anthropic's own host gets a placeholder.
            api_key=_outbound_key(settings),
            # None means the SDK default. Set ANTHROPIC_BASE_URL to route
            # through a proxy without changing a line of this.
            base_url=settings.anthropic_base_url or None,
        )
        self._index = None

    @property
    def namespace(self) -> str:
        return self._settings.pinecone_namespace

    @property
    def index(self):
        if self._index is None:
            self._index = Pinecone(
                api_key=self._settings.pinecone_api_key
            ).Index(self._settings.pinecone_index)
        return self._index

    # -- ingestion ----------------------------------------------------------
    async def ingest(self, episodes: list[Episode]) -> int:
        rows: list[dict] = []
        for ep in episodes:
            for start_t, text, stamped in _windows(
                ep.segments, self._settings.chunk_max_chars, overlap_segments=2
            ):
                rows.append({
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "url": ep.url,
                    "start_seconds": start_t,
                    "text": text,
                    # Never embedded, never shown to a reader — see _windows.
                    "text_ts": stamped,
                    # Pinecone metadata rejects None, so undated episodes
                    # omit the key entirely rather than storing a null.
                    **({"published_at": ep.published_at}
                       if ep.published_at else {}),
                })
        if not rows:
            return 0

        embeddings = await embed_texts(
            self._voyage,
            # Embed the title with the window. This matters more here than
            # it did on Market Bubble: an MCG title names the project, and
            # the project name is often barely spoken aloud after the
            # intro. Without the title, "what does Umia do" has to match on
            # topic words alone. The stored excerpt stays the transcript,
            # so a reader still sees only what was said.
            [f"{r['title']}\n\n{r['text']}" for r in rows],
            model=self._settings.voyage_model,
            dimension=self._settings.embedding_dimension,
            input_type="document",
        )

        vectors = [
            {
                # Deterministic, so re-running an episode overwrites its own
                # rows instead of duplicating them. That is what makes the
                # ingest safe to interrupt and resume.
                "id": hashlib.sha256(
                    f"{r['episode_id']}:{r['start_seconds']}".encode()
                ).hexdigest()[:32],
                "values": emb,
                "metadata": r,
            }
            for r, emb in zip(rows, embeddings, strict=True)
        ]

        def _upsert() -> None:
            for start in range(0, len(vectors), 100):
                self.index.upsert(
                    vectors=vectors[start:start + 100], namespace=self.namespace
                )

        # Bound the write: the Pinecone client has no read timeout, so a dead
        # socket would hang the ingest indefinitely. On timeout this raises,
        # and the deterministic ids above make the retry safe.
        await asyncio.wait_for(
            asyncio.to_thread(_upsert),
            timeout=self._settings.pinecone_write_timeout_seconds,
        )
        logger.info("Indexed %d windows from %d episode(s)",
                    len(vectors), len(episodes))
        return len(vectors)

    # -- search -------------------------------------------------------------
    async def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self._settings.retrieval_top_k
        vector = await embed_query(
            self._voyage, query,
            model=self._settings.voyage_model,
            dimension=self._settings.embedding_dimension,
        )

        # Pull a wider candidate set when reranking is on; the reranker
        # narrows it back to top_k by actual relevance.
        fetch_k = (max(self._settings.rerank_candidates, top_k)
                   if self._settings.rerank_model else top_k)

        def _query():
            return self.index.query(
                vector=vector, top_k=fetch_k, namespace=self.namespace,
                include_metadata=True,
            )

        response = await asyncio.wait_for(
            asyncio.to_thread(_query),
            timeout=self._settings.pinecone_read_timeout_seconds,
        )

        hits: list[Hit] = []
        for match in response.matches:
            if match.score < self._settings.retrieval_min_score:
                continue
            md = match.metadata or {}
            start = float(md.get("start_seconds", 0))
            hits.append(Hit(
                episode_id=md.get("episode_id", ""),
                title=md.get("title", ""),
                start_seconds=start,
                timestamp=_timestamp(start),
                deep_link=_deep_link(md.get("url", ""), start),
                text=md.get("text", ""),
                # Falls back to the plain text so vectors written before
                # text_ts existed still answer, just with a coarser citation.
                text_ts=md.get("text_ts") or md.get("text", ""),
                published_at=md.get("published_at"),
                score=match.score,
            ))

        if self._settings.rerank_model and len(hits) > top_k:
            # Title first, same as at ingest. Reranking the transcript alone
            # throws away the title signal the embedding just used, so an
            # episode found *because* of its title gets demoted by the stage
            # meant to improve the ordering.
            docs = [f"{h.title}\n\n{h.text}" for h in hits]
            order = await rerank_order(
                self._voyage, query, docs,
                top_k=top_k, model=self._settings.rerank_model,
            )
            if order is not None:
                hits = [hits[i] for i in order]
        return hits[:top_k]

    @staticmethod
    def _format(hits: list[Hit]) -> str:
        if not hits:
            return "<excerpts>\n(nothing indexed matched this query)\n</excerpts>"
        # Chronological, so a topic reads in the order it was discussed.
        # Relevance order is what the reader sees in the hit list; the model
        # needs the timeline. Undated excerpts sort last rather than
        # inventing a position for them.
        hits = sorted(hits, key=lambda h: (h.published_at is None,
                                           h.published_at or "", h.start_seconds))
        # Transcript text and titles are untrusted third-party captions.
        # XML-escape them so a crafted window cannot forge a closing
        # </excerpt> tag and break out of the region the system prompt
        # treats as grounding.
        blocks = [
            f"<excerpt episode={quoteattr(h.title)} at={quoteattr(h.timestamp)}"
            + (f" aired={quoteattr(h.published_at)}" if h.published_at else "")
            + f">\n{escape(h.text_ts or h.text)}\n</excerpt>"
            for h in hits
        ]
        return "<excerpts>\n" + "\n\n".join(blocks) + "\n</excerpts>"

    def _build_request(self, query: str, hits: list[Hit]) -> dict:
        return {
            "model": self._settings.search_model,
            "max_tokens": self._settings.search_max_tokens,
            "system": [
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._format(hits)},
                    {"type": "text", "text": query,
                     "cache_control": {"type": "ephemeral"}},
                ],
            }],
        }

    async def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        hits = await self.retrieve(query, top_k)
        client = self._anthropic.with_options(
            timeout=self._settings.search_timeout_seconds
        )
        response = await client.beta.messages.create(
            **self._build_request(query, hits)
        )
        if response.stop_reason == "refusal":
            return SearchResponse(answer=REFUSAL_ANSWER, hits=[],
                                  model=response.model, refused=True)
        answer = "".join(b.text for b in response.content if b.type == "text")
        return SearchResponse(answer=answer, hits=hits, model=response.model)
