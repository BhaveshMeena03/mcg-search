"""Configuration for MCG search.

Everything comes from the environment (or a local .env), so no key is ever
in the source. The defaults below are the ones the Market Bubble index was
tuned to over ~1,100 tests — they are a starting point here, not a result,
and the ones worth re-measuring on this archive are marked.
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
    # Haiku 4.5 is what Market Bubble search answers on. The job is to
    # summarise passages that retrieval already found, not to reason from
    # scratch, and the small model does that at ~$0.004 a question.
    search_model: str = "claude-haiku-4-5"
    search_max_tokens: int = 1024
    search_timeout_seconds: float = 45.0
    # Set ANTHROPIC_BASE_URL to route through a proxy without touching code.
    anthropic_base_url: str | None = None

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
    # MCG gets its OWN index. See the guard in scripts/ingest_episodes.py —
    # pointing this at bullpen-concierge is the one mistake that could reach
    # the live Market Bubble search, so it is refused rather than documented.
    pinecone_index: str = "mcg-search"
    pinecone_namespace: str = "mcg"
    pinecone_write_timeout_seconds: float = 60.0
    pinecone_read_timeout_seconds: float = 20.0

    # --- Retrieval ---------------------------------------------------------
    # Worth re-measuring on this archive. MCG is one project per episode,
    # where Market Bubble is forty topics per broadcast, so the right number
    # of passages to feed an answer is probably not the same. Starting with
    # what is known to work rather than with a guess.
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.05
    chunk_max_chars: int = 2400

    # --- Channel -----------------------------------------------------------
    youtube_channel: str = "https://www.youtube.com/@MCG_live/videos"


@lru_cache
def get_settings() -> Settings:
    return Settings()
