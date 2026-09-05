"""Configuration for MCG search.

Everything comes from the environment (or a local .env), so no key is ever
in the source. The defaults below were tuned on a comparable archive and
are a starting point here, not a result; the ones worth re-measuring on
this one are marked.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Anthropic ---------------------------------------------------------
    # Optional, because it is genuinely not needed when routing through a
    # proxy that authenticates on its own URL token — usepod does, and it
    # documents this key as ignored. Keeping it required would force a
    # live Anthropic credential to sit in .env for no reason, which is the
    # opposite of what you want: the safest place for a secret you do not
    # need is nowhere.
    #
    # Set it only to talk to api.anthropic.com directly. _outbound_key()
    # in app/search.py decides which of the two is in play, and never
    # sends a real key anywhere but Anthropic's own host.
    anthropic_api_key: str = ""
    # Haiku 4.5. The job is to summarise passages retrieval already
    # found, not to reason from scratch, and the small model does that at
    # about $0.004 a question.
    search_model: str = "claude-haiku-4-5"
    search_max_tokens: int = 1024
    # Measured, not guessed. Through the proxy the same question took
    # between 6.4 and 44.5 seconds across this session, so the old 45s
    # left half a second of margin against the worst run actually seen.
    # With no Anthropic fallback configured a timeout is not a slow
    # answer, it is no answer, so the ceiling sits at roughly twice the
    # worst case. The page streams citations at ~3s, which is what makes
    # the tail bearable.
    search_timeout_seconds: float = 90.0
    # Set ANTHROPIC_BASE_URL to route through a proxy without touching code.
    anthropic_base_url: str | None = None

    # Which upstream providers the proxy may use, in preference order.
    # Empty means "let it route", which optimises for price and is why the
    # source can change between two identical requests: this archive went
    # from a relay doing 70 output tok/s to one doing 5.9 overnight, with
    # no change here.
    #
    # "anthropic" pins to first-party Claude. Note what that costs: a pin
    # skips the marketplace entirely, so the price is the centralized tier
    # ($0.80/$4.00 per M) rather than the marketplace one ($0.40/$2.00).
    # Still under Anthropic direct at $1.00/$5.00, and predictable, which
    # is the thing worth buying for a page someone is waiting on.
    #
    # A pin is a HARD constraint: unsatisfiable pins 503 rather than
    # quietly falling back, so this and the Anthropic fallback in
    # app/search.py do different jobs and both earn their place.
    usepod_providers: str = ""

    # Whether the page's answer arrives token by token.
    #
    # Off, because measured through the proxy streaming is strictly worse
    # than not streaming. Pinned to Anthropic the whole non-streamed call
    # returns in 4.4s, while the streamed one takes 18.9s just to reach
    # the FIRST token and arrives in 8-10 lumps rather than the ~90 a
    # real stream produces. The proxy is buffering.
    #
    # So the page sends citations as soon as retrieval finishes -- about
    # a second -- and then the finished answer in one piece a few seconds
    # later. That is faster to a complete answer than watching it trickle.
    # Turn this back on if talking to Anthropic directly, where streaming
    # is genuinely incremental.
    stream_answers: bool = False

    # --- Voyage (embeddings + reranker) ------------------------------------
    voyage_api_key: str
    voyage_model: str = "voyage-3.5"
    embedding_dimension: int = 1024
    # Only needed on Voyage's free tier (3 requests/minute). On a paid key
    # this adds 21s per batch for nothing.
    voyage_request_gap_seconds: float = 0.0
    rerank_model: str | None = "rerank-2.5-lite"
    rerank_candidates: int = 50

    # --- Pinecone ----------------------------------------------------------
    pinecone_api_key: str
    # MCG gets its own index. See check_target in
    # scripts/ingest_episodes.py for how to mark others off-limits.
    pinecone_index: str = "mcg-search"
    pinecone_namespace: str = "mcg"

    # Indexes and namespaces the ingest must refuse to write to, comma
    # separated. Empty by default.
    #
    # This exists because one Pinecone account can hold several
    # unrelated projects, and a scheduled job with a wrong .env is the
    # one mistake here that reaches something else's data. It is
    # configuration rather than a constant in the source so this repo
    # carries no knowledge of what else happens to share the account —
    # a deployment's neighbours are its operator's business, not
    # something to publish in a codebase.
    protected_indexes: str = ""
    protected_namespaces: str = ""
    pinecone_write_timeout_seconds: float = 60.0
    pinecone_read_timeout_seconds: float = 20.0

    # --- Retrieval ---------------------------------------------------------
    # Worth re-measuring on this archive. MCG is one project per episode,
    # rather than forty topics per broadcast, so the right number of
    # passages to feed an answer is probably not what it is elsewhere.
    # Starting from a known-good number rather than a guess.
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.05
    chunk_max_chars: int = 2400

    # --- Channel -----------------------------------------------------------
    youtube_channel: str = "https://www.youtube.com/@MCG_live/videos"


@lru_cache
def get_settings() -> Settings:
    return Settings()
