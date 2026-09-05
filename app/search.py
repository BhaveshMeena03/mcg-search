"""MCG episode search.

Ingests timestamped episode transcripts, embeds windows of consecutive
segments (keeping each window's start time), and answers questions with an
answer plus citations that deep-link to the exact second in the video.

The parts that took the measuring are the windowing, the separate
embedded vs stamped text, and the prompt rules — each written after a
wrong answer went out in public. Everything here assumes MCG's format:
one host interviewing one project per episode.
"""

import asyncio
import hashlib
import logging
import re
import time
from urllib.parse import urlparse
from xml.sax.saxutils import escape, quoteattr

import anthropic
import voyageai
from anthropic import AsyncAnthropic
from pinecone import Pinecone

from .config import get_settings
from .embeddings import embed_query, embed_texts, rerank_order
from .overlap import is_duplicate
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
2a. When the same thing is covered by a dedicated episode AND by an \
excerpt marked kind="live stream", cite the episode. The streams are \
four-hour broadcasts and the episodes are the interviews cut out of \
them, so both are accurate and they are not equally useful: asked what \
Scopl does, the answer cited 2:23:48 in a market update when the Scopl \
episode explains it at 6:59. A reader pressing the first lands two and \
a half hours into something else. Cite a stream when it is the only \
source, or when it genuinely adds something the episode does not — \
market commentary about a project usually is not in its own episode.
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
5b-i. A SPELLING DIFFERENCE IS NEVER A REASON TO SAY YOU COULD NOT FIND \
SOMETHING. Asked "what is AVICI", the excerpts were the right episodes and \
the answer opened "I couldn't find that in the episodes I've indexed. The \
project is called Avichi, not AVICI" — and then described the project \
correctly anyway. Every part of that is wrong: the archive did contain it, \
the reader stops at the first sentence, and "Avichi" was the caption's \
guess. If you can describe the project, you found it. Answer.
5b-ii. Never state what an episode title says. You are given titles as \
the `episode` attribute; do not claim a title uses a spelling, because \
that claim was made about AVICI and was false — the titles read "$AVICI" \
and "AviciMoney". Describe what was SAID, and let the title stand as it \
is.
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
6a. When the QUESTION asks you to judge, compare, rank or pick for \
investment purposes, say you can't do that FIRST, then report what \
speakers actually said. Both halves matter and the order matters. \
Measured inconsistency: "what should I buy right now" and "should I sell \
my Meteora position" both opened with a clean decline and then gave the \
archive's content, which is right. But "is BIT10 better than holding \
BTC" opened "According to a 10-year backtest... the index beat Bitcoin", \
and "which of these tokens has the best risk reward" went straight to \
quoting a founder's claim. Those report attributed facts and are still \
wrong, because a comparison question answered with a favourable number \
and no decline reads as the tool endorsing it, whoever is quoted. \
Reporting that a founder is bullish about their own project is always \
fine; leading with it, when the question asked you to pick, is not.
6a-i. "YOUR pick", "your favourite", "what would you buy" is addressed \
to YOU, not to a guest. Asked "what is your highest conviction pick" the \
answer named a guest's number-one coin, reading "your" as the speaker's. \
Attributed and dated, and still the wrong shape: a bot that returns a \
coin name to that question is screenshotted as having picked it. Decline \
first, then say what guests named if it is useful.
6c. RELAYING A MARKET CALL IS STILL GIVING ONE. The archive now \
includes four-hour live market broadcasts, where the hosts say what they \
are buying and where they think prices go. Attribution does not make \
that safe to pass on: "they are advocating buying the dip, and called \
Pawns obviously a buy the dip coin" is a coin recommendation with a \
citation attached, and it is read as one. That answer was produced by \
this tool and is why this rule exists.\
 So before answering, look at what the answer will SAY, not at how the \
question was phrased. If it amounts to buy this, sell this, or this \
will reach that price — whoever said it first — open by declining, then \
report it as commentary: who said it, when, and that it was their view \
at that moment rather than a call this tool is passing on. Never lead \
with the position. Never name an asset as a buy or a sell in your first \
sentence.
6d. What IS fine, and should not be refused: what a project does, what \
was claimed about it, what a founder said about their own company, what \
topics were covered, what happened in a market and when. "What did they \
say about the Fed decision" is reporting. "What are they buying" is not, \
however it is worded — and the difference is the answer, not the \
question.
6b. A decline is not the denial rule 1a forbids. Rule 1a is about \
claiming you could not FIND something you did find. Saying "I can't tell \
you what to buy" and then citing what was said is the correct shape, and \
you should still open the citation half with what WAS said rather than \
with what was not.
7. Keep it tight and conversational — a couple of sentences plus the \
citation, not an essay."""


def _timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _deep_link(url: str, seconds: float) -> str:
    """A watch URL that jumps to the moment.

    YouTube only, because the whole channel is.
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
    windows: list[tuple[float, str, list[float]]] = []
    i = 0
    n = len(segments)
    while i < n:
        start_t = segments[i].t
        parts: list[str] = []
        times: list[float] = []
        length = 0
        j = i
        while j < n and length + len(segments[j].text) + 1 <= max_chars:
            parts.append(segments[j].text)
            times.append(segments[j].t)
            length += len(segments[j].text) + 1
            j += 1
        if j == i:  # single segment longer than max_chars — take it whole
            parts.append(segments[i].text[:max_chars])
            times.append(segments[i].t)
            j = i + 1
        windows.append((start_t, "\n".join(parts), times))
        if j >= n:
            break
        i = max(j - overlap_segments, i + 1)
    return windows


def pack_times(line_times: list[float]) -> str:
    """Line start times as a compact string, e.g. "0,7,14,21".

    Pinecone metadata takes a string, number, boolean or list of
    strings — a list of floats is rejected with a 400. A comma-joined
    string of whole seconds is both legal and the smallest form, and
    nothing is lost: _timestamp() truncates to the second anyway, and
    the deep link uses the window's own start_seconds, which is stored
    separately as a number.
    """
    return ",".join(str(int(t)) for t in line_times)


def stamp(text: str, line_times) -> str:
    """Rebuild the per-line timestamped copy from the text and its times.

    This used to be stored alongside the text as `text_ts` — the same
    words a second time, with "[12:34] " in front of each line. Every
    query pulls 50 rerank candidates, so that duplicate was travelling
    on every search: ~234KB per question, against a metadata payload
    that is otherwise half the size.

    Retrieval and reranking never read the stamped copy. Only the answer
    does. So store the line start times, about a hundred bytes, and
    rebuild it here.

    Falls back to the plain text when the times are missing or do not
    line up, which is what makes vectors written before this change keep
    working.
    """
    if not line_times:
        return text
    if isinstance(line_times, str):
        try:
            line_times = [float(x) for x in line_times.split(",") if x]
        except ValueError:
            return text
    lines = text.split("\n")
    if len(lines) != len(line_times):
        return text
    return "\n".join(f"[{_timestamp(t)}] {line}"
                     for t, line in zip(line_times, lines, strict=True))


PLACEHOLDER_KEY = "unused-auth-is-in-the-base-url"

# Questions whose honest answer is a trading instruction.
#
# The archive now holds four-hour market broadcasts where hosts say what
# they are buying. Attribution does not make that safe to relay: asked
# whether they are saying to buy the dip, this tool answered "the hosts
# are advocating a buy the dip strategy" and named a coin. Cited, true,
# and indistinguishable from a recommendation to anybody screenshotting
# it.
#
# The prompt was told not to. It complied about five times in six —
# measured, on identical repeated runs — which is fine for style and not
# fine for this. So the decline is prepended in code, where it cannot be
# argued out of, and the answer still follows it.
_MARKET_CALL = re.compile(
    r"\b(buy|buying|sell|selling|short|long|ape|aping)\b.{0,40}\b"
    r"(dip|now|this|it|in|into)\b"
    r"|\bbullish\b|\bbearish\b|\bprice target\b|\bbetting on\b"
    r"|\bwhat (should|would) (i|you) (buy|sell|invest|allocate)"
    r"|\bworth (buying|selling|investing)\b"
    r"|\b(moon|pump|100x|10x)\b",
    re.IGNORECASE,
)

MARKET_CALL_PREFIX = (
    "I can't tell you what to buy or sell, and anything the hosts say "
    "about a trade is their view at that moment, not a recommendation "
    "from this tool. What was actually said:"
)


def is_market_call(question: str) -> bool:
    return bool(_MARKET_CALL.search(question or ""))


def _already_declines(answer: str) -> bool:
    """Has the answer opened by declining on its own?

    Only the opening counts. An answer that names a coin and adds a
    caveat at the end has already been read by then.
    """
    opening = " ".join(
        re.split(r"(?<=[.!?])\s", (answer or "").strip())[:1]
    ).lower()
    return any(p in opening for p in (
        "i can't", "i cannot", "i won't", "i will not",
        "couldn't find", "could not find", "not investment advice",
    ))

# Below this, a "question" carries no subject to search for. Three
# characters keeps real ones: tickers get asked about as "$UP", and "wen"
# is a whole question on crypto X.
MIN_QUERY_CHARS = 3

EMPTY_QUERY_ANSWER = (
    "Ask me something about the MCG episodes — a project name, or what "
    "someone said about a topic."
)

# Faults where trying again is cheap and likely to work: the connection
# never established, or the far end returned a 5xx. These fail in
# milliseconds, so a retry costs almost nothing.
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)

# A timeout is different and must not be lumped in with those. It means
# the proxy accepted the request and is being slow, so retrying waits the
# whole ceiling again — at 90s that is three minutes of a reader watching
# a cursor before they get nothing. Retry the cheap failures; fail fast
# on the expensive one.
_SLOW = (
    anthropic.APITimeoutError,
    asyncio.TimeoutError,
)

# Anything that should hand over to the fallback, when one is configured.
_TRANSIENT = _RETRYABLE + _SLOW
ANSWER_RETRIES = 3

# A proxy that just failed should cost one slow answer, not one per
# visitor for as long as it stays down. Long enough that an outage is not
# re-tested by everybody who shows up, short enough that recovery needs
# no deploy.
class StreamAlreadyStarted(Exception):
    """The proxy died mid-answer, after text had already been sent.

    Distinct from a clean failure because the recovery differs: before
    the first token a fallback is invisible to the reader, after it a
    second attempt would print the answer twice.
    """


PROXY_COOLDOWN_SECONDS = 300

# Named, and that is not decoration. The SDK reads ANTHROPIC_BASE_URL
# from the environment when it is not told otherwise, which is exactly
# how the proxy gets configured — so a fallback client built with only a
# key inherits the proxy and quietly becomes a second route to the thing
# that just failed.
ANTHROPIC_DIRECT_URL = "https://api.anthropic.com"

# Hosts the real Anthropic credential may be sent to. Anything else is a
# third party, however trustworthy, and gets the placeholder instead.
_ANTHROPIC_HOSTS = ("api.anthropic.com",)


def _redact_proxy(text: str) -> str:
    """Hide a proxy token in anything about to be logged.

    usepod authenticates on a token in the URL PATH, so it rides inside
    every URL the SDK builds and turns up in exception text. An error
    logged raw would write a live credential into the server log.
    """
    return re.sub(r"(/proxy/)[^/\s]+", r"\1<token>", text or "")


class Misconfigured(RuntimeError):
    """A settings combination that cannot answer anything.

    Raised at startup rather than per request. The failure it replaces
    was a TypeError about authentication methods on every single
    question, which says nothing about the cause.
    """


def check_credentials(settings) -> None:
    """Fail at boot when nothing can authenticate.

    Going direct to Anthropic needs a key. This happened for real: the
    shell exported ANTHROPIC_BASE_URL=https://api.anthropic.com while
    the key had been removed, so every request raised an SDK TypeError
    and the page just said the answer stopped early.
    """
    base = (settings.anthropic_base_url or "").strip()
    host = (urlparse(base).hostname or "").lower() if base else ""
    going_direct = not base or host in _ANTHROPIC_HOSTS
    if going_direct and not settings.anthropic_api_key:
        raise Misconfigured(
            "No way to reach a model. ANTHROPIC_BASE_URL is "
            f"{base or 'unset'}, which means Anthropic directly, but "
            "ANTHROPIC_API_KEY is empty. Either set the key, or point "
            "ANTHROPIC_BASE_URL at the proxy. Note a shell variable "
            "overrides .env, which is how this usually happens."
        )


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
        # Anthropic direct, held ready, and only when the client above is
        # NOT already that. A proxy can run out of balance, have its token
        # revoked, or get routed to a provider that stops answering — and
        # every one of those looks like the site being down to somebody
        # typing a question. Measured here: the route silently moved from
        # a relay doing 70 tok/s to one doing 5.9, and a trivial request
        # stopped returning inside two minutes. Nothing in this repo
        # caused it and nothing in this repo could have prevented it.
        #
        # This client carries the real key, which is exactly why the one
        # above does not.
        self._fallback = None
        if settings.anthropic_base_url and settings.anthropic_api_key:
            self._fallback = AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                base_url=ANTHROPIC_DIRECT_URL,
            )
        # When the proxy last failed. Inside the cooldown every request
        # goes straight to Anthropic rather than paying the timeout again.
        self._proxy_failed_at = 0.0
        self._index = None

    def _llm(self):
        """The client to try, and whether a fallback is still held back.

        Inside the cooldown this hands back Anthropic directly, so an
        outage costs one slow answer rather than one per visitor for as
        long as it lasts.
        """
        if self._fallback is None:
            return self._anthropic, False
        cooling = (time.monotonic() - self._proxy_failed_at
                   < PROXY_COOLDOWN_SECONDS)
        if cooling:
            return self._fallback, False
        return self._anthropic, True

    def _proxy_broke(self, exc: Exception) -> None:
        self._proxy_failed_at = time.monotonic()
        logger.error(
            "the model proxy failed (%s) — answering on Anthropic direct, "
            "and skipping the proxy for %ds",
            _redact_proxy(str(exc))[:200], PROXY_COOLDOWN_SECONDS)

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
    async def ingest(self, episodes: list[Episode],
                     reference: set[str] | None = None) -> int:
        """Index episodes, skipping stream windows already in a clip.

        `reference` is the shingle set of the clips a stream might
        contain — see app/overlap.py. Measured on a real broadcast, 43%
        of its windows are the interviews that were cut out of it and
        published separately, and indexing those again would let the
        same conversation be retrieved twice, with the worse copy
        sometimes winning: a citation into a 42-minute episode about one
        project beats the identical words two hours forty into a
        four-hour stream.

        None means index everything, which is what a clip wants.
        """
        rows: list[dict] = []
        skipped = 0
        for ep in episodes:
            for start_t, text, line_times in _windows(
                ep.segments, self._settings.chunk_max_chars, overlap_segments=2
            ):
                if (reference and ep.format == "stream"
                        and is_duplicate(text, reference)):
                    skipped += 1
                    continue
                rows.append({
                    "format": ep.format,
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "url": ep.url,
                    "start_seconds": start_t,
                    "text": text,
                    # The start time of each line, so the timestamped copy
                    # can be rebuilt at answer time. This replaced storing
                    # the whole stamped string: it was the same words a
                    # second time, and every query drags 50 rerank
                    # candidates across the wire. ~100 bytes instead of
                    # ~2400. See stamp().
                    "line_times": pack_times(line_times),
                    # Pinecone metadata rejects None, so undated episodes
                    # omit the key entirely rather than storing a null.
                    **({"published_at": ep.published_at}
                       if ep.published_at else {}),
                })
        if skipped:
            logger.info("skipped %d window(s) already published as a clip",
                        skipped)
        if not rows:
            return 0

        embeddings = await embed_texts(
            self._voyage,
            # Embed the title with the window. This matters more here than
            # it does elsewhere: an MCG title names the project, and
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
                # Rebuilt from the line times. Vectors written before this
                # change still carry text_ts, so prefer that when present
                # and neither copy needs a re-ingest to keep working.
                text_ts=(md.get("text_ts")
                         or stamp(md.get("text", ""), md.get("line_times"))),
                published_at=md.get("published_at"),
                format=md.get("format", "interview"),
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
            # So rule 2a can tell a four-hour broadcast from the clip cut
            # out of it. Both are citable; one is a far better place to
            # send a reader.
            + (' kind="live stream"' if h.format == "stream" else "")
            + f">\n{escape(h.text_ts or h.text)}\n</excerpt>"
            for h in hits
        ]
        return "<excerpts>\n" + "\n\n".join(blocks) + "\n</excerpts>"

    def _proxy_headers(self, going_direct: bool = False) -> dict:
        """X-Pod-Providers, when a pin is configured and we're proxied.

        Never sent to api.anthropic.com: an X-Pod-* header means nothing
        there, and sending it would leak which proxy this app uses.
        """
        pin = (self._settings.usepod_providers or "").strip()
        if going_direct or not pin or not self._settings.anthropic_base_url:
            return {}
        return {"X-Pod-Providers": pin}

    def _proxy_headers_kwargs(self, going_direct: bool = False) -> dict:
        """The pin as create() kwargs, or nothing. Lets callers outside
        this class — the summariser — route the same way a search does
        without reaching into request internals."""
        headers = self._proxy_headers(going_direct)
        return {"extra_headers": headers} if headers else {}

    def _build_request(self, query: str, hits: list[Hit],
                       going_direct: bool = False) -> dict:
        headers = self._proxy_headers(going_direct)
        return {
            **({"extra_headers": headers} if headers else {}),
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

    async def _answer(self, query: str, hits: list[Hit]):
        """One model call, retried on a dropped connection.

        Retrieval already survives a transient network fault; the answer
        call did not, so a blip after the expensive embedding+rerank work
        threw the whole question away. One TimeoutError in ~320 graded
        calls — 0.3%, invisible in a CLI and a visible dead reply on a
        public bot.

        Only transport faults retry. A refusal, a bad request or an auth
        error is a real answer about the request and repeating it just
        spends money to fail again.
        """
        chosen, can_fall_back = self._llm()
        # True when the client in hand is Anthropic itself, either
        # because no proxy is configured or because we already fell over.
        going_direct = chosen is self._fallback
        client = chosen.with_options(
            timeout=self._settings.search_timeout_seconds
        )
        last: Exception | None = None
        for attempt in range(1, ANSWER_RETRIES + 1):
            try:
                return await client.beta.messages.create(
                    **self._build_request(query, hits, going_direct)
                )
            except _TRANSIENT as exc:
                last = exc
                # One failure on the proxy is enough to stop using it.
                # Retrying a route that has already timed out just pays
                # the timeout again, and the way back is right there.
                if can_fall_back:
                    self._proxy_broke(exc)
                    client = self._fallback.with_options(
                        timeout=self._settings.search_timeout_seconds
                    )
                    going_direct = True
                    can_fall_back = False
                    continue
                # A timeout already cost the full ceiling. Waiting it
                # again is worse for the reader than admitting defeat.
                if isinstance(exc, _SLOW) or attempt == ANSWER_RETRIES:
                    break
                wait = 2 ** (attempt - 1)
                logger.warning("answer call failed (%s), retry %d/%d in %ds",
                               type(exc).__name__, attempt, ANSWER_RETRIES, wait)
                await asyncio.sleep(wait)
        raise last                                            # type: ignore[misc]

    async def answer_stream(self, query: str, hits: list[Hit]):
        """Yield answer text deltas for already-retrieved hits.

        Streaming is not a nicety here. Measured through the proxy, a
        single answer takes between 9.5 and 43 seconds on identical
        input — a 4.5x spread. A page that shows nothing for 43 seconds
        looks broken; the same wait with words arriving reads as
        thinking. The caller retrieves first so it can paint the
        citations before the first token lands.

        No retry wrapper: once bytes have been sent to the browser a
        retry would duplicate the answer. The caller handles a mid-stream
        failure by ending the stream honestly.
        """
        chosen, can_fall_back = self._llm()
        try:
            async for chunk in self._stream_once(
                chosen, query, hits, going_direct=chosen is self._fallback
            ):
                yield chunk
            return
        except _TRANSIENT as exc:
            # Only safe to fail over before any text has been sent. Once
            # bytes are on the wire a second attempt would repeat the
            # answer from the top, so _stream_once raises Started for
            # anything that breaks mid-answer and it lands below.
            if not can_fall_back:
                raise
            self._proxy_broke(exc)
        async for chunk in self._stream_once(self._fallback, query, hits,
                                            going_direct=True):
            yield chunk

    async def _stream_once(self, client, query: str, hits: list[Hit],
                           going_direct: bool = False):
        started = False
        try:
            bound = client.with_options(
                timeout=self._settings.search_timeout_seconds
            )
            async with bound.beta.messages.stream(
                **self._build_request(query, hits, going_direct)
            ) as stream:
                async for text in stream.text_stream:
                    started = True
                    yield text
                final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\x00REFUSAL\x00"
        except _TRANSIENT:
            if started:
                # Half an answer is already rendered. Ending here is
                # honest; restarting would print it twice.
                raise StreamAlreadyStarted from None
            raise

    async def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        # A query too short to carry meaning retrieves whatever happens to
        # be nearest and asks the model to explain it. "?" produced an
        # answer referring to "the episode title you're asking about" when
        # no title had been asked about — incoherent rather than harmful,
        # but on a public bot every reply is published. Cheaper and more
        # honest to say nothing was asked.
        if len(query.strip()) < MIN_QUERY_CHARS:
            return SearchResponse(answer=EMPTY_QUERY_ANSWER, hits=[],
                                  model=self._settings.search_model)

        hits = await self.retrieve(query, top_k)
        response = await self._answer(query, hits)
        if response.stop_reason == "refusal":
            return SearchResponse(answer=REFUSAL_ANSWER, hits=[],
                                  model=response.model, refused=True)
        answer = "".join(b.text for b in response.content if b.type == "text")
        # Deterministic, because the prompt rule is not. See _MARKET_CALL.
        if is_market_call(query) and not _already_declines(answer):
            answer = f"{MARKET_CALL_PREFIX}\n\n{answer}"
        return SearchResponse(answer=answer, hits=hits, model=response.model)
