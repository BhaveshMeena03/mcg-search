"""Index whatever is in data/episodes.json.

    .venv/bin/python scripts/ingest_episodes.py
    .venv/bin/python scripts/ingest_episodes.py --only VIDEO_ID
    .venv/bin/python scripts/ingest_episodes.py --force     # re-embed all

Incremental by design. Episodes already in the index are skipped, so this
is safe to re-run after adding episodes and costs nothing for the ones
already done. Rebuilding the whole namespace from scratch would be an
hour of embeddings to get back where you already were.

Run after scripts/fetch_episodes.py.
"""

import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.search import MCGIndex  # noqa: E402

DATA = ROOT / "data" / "episodes.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def log(msg: str) -> None:
    print(msg, flush=True)  # visible immediately even when redirected


def _protected(raw: str) -> set[str]:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


def check_target(settings) -> None:
    """Refuse to write to an index or namespace marked off-limits.

    One Pinecone account can hold several unrelated projects, and a
    scheduled job with a wrong .env is the one mistake here that reaches
    something else's data. Set PROTECTED_INDEXES / PROTECTED_NAMESPACES
    to whatever this deployment must never touch; both are empty by
    default, so a fresh install has nothing to trip over.
    """
    indexes = _protected(getattr(settings, "protected_indexes", ""))
    namespaces = _protected(getattr(settings, "protected_namespaces", ""))
    if settings.pinecone_index in indexes:
        raise SystemExit(
            f"REFUSING TO RUN: PINECONE_INDEX is "
            f"'{settings.pinecone_index}', which PROTECTED_INDEXES marks "
            f"as off-limits. Fix PINECONE_INDEX in .env."
        )
    if settings.pinecone_namespace in namespaces:
        raise SystemExit(
            f"REFUSING TO RUN: PINECONE_NAMESPACE is "
            f"'{settings.pinecone_namespace}', which PROTECTED_NAMESPACES "
            f"marks as off-limits. Fix PINECONE_NAMESPACE in .env."
        )


def ensure_index(settings) -> None:
    """Create the MCG index if it is not there yet.

    Serverless, same cloud/region/dimension as the existing indexes so the
    same Voyage embeddings fit. Creating one does not touch any other index
    in the account.
    """
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = {i.name for i in pc.list_indexes()}
    if settings.pinecone_index in existing:
        return
    log(f"creating Pinecone index '{settings.pinecone_index}' "
        f"(dim={settings.embedding_dimension}, cosine, aws/us-east-1)")
    pc.create_index(
        name=settings.pinecone_index,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Creation is async on Pinecone's side; wait for it to accept writes.
    while not pc.describe_index(settings.pinecone_index).status.get("ready"):
        time.sleep(2)
    log("index ready")


def first_window_id(ep: Episode) -> str:
    """The vector id of an episode's first window.

    Ids are sha256(episode_id:start_seconds), and the first window always
    starts at the first segment's timestamp — so this is computable without
    building every window, and its presence means the episode is indexed.
    """
    start = ep.segments[0].t if ep.segments else 0.0
    return hashlib.sha256(
        f"{ep.episode_id}:{start}".encode()
    ).hexdigest()[:32]


async def main(argv: list[str]) -> int:
    settings = get_settings()
    check_target(settings)

    force = "--force" in argv
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]

    if not DATA.exists():
        log("data/episodes.json is missing — run fetch_episodes.py first.")
        return 1
    episodes = [Episode(**e) for e in json.loads(DATA.read_text())]
    if only:
        episodes = [e for e in episodes if e.episode_id == only]
    if not episodes:
        log("nothing to ingest.")
        return 1

    ensure_index(settings)
    idx = MCGIndex()
    log(f"target: index='{settings.pinecone_index}' "
        f"namespace='{idx.namespace}' — {len(episodes)} episode(s) on file")

    # Ask Pinecone which are already in, rather than keeping a state file
    # that can disagree with reality.
    todo = episodes
    if not force:
        ids = {first_window_id(e): e for e in episodes}
        present: set[str] = set()
        all_ids = list(ids)
        for start in range(0, len(all_ids), 100):
            chunk = all_ids[start:start + 100]

            def _fetch(batch=chunk):
                return idx.index.fetch(ids=batch, namespace=idx.namespace)

            # Bounded like every other Pinecone call. The client has no
            # read timeout of its own, so a half-open socket here would
            # hang before a single episode had been indexed.
            try:
                got = await asyncio.wait_for(
                    asyncio.to_thread(_fetch),
                    timeout=settings.pinecone_read_timeout_seconds,
                )
            except Exception as exc:                          # noqa: BLE001
                # Resume is an optimisation, not correctness: the vector
                # ids are deterministic, so re-indexing an episode
                # overwrites its own rows. Failing to check costs money,
                # not data, so say so and do the work rather than stop.
                log(f"resume check failed ({exc}) — indexing everything")
                present = set()
                break
            present |= set(getattr(got, "vectors", {}) or {})
        else:
            todo = [e for vid, e in ids.items() if vid not in present]
        log(f"{len(episodes) - len(todo)} already indexed, {len(todo)} to do")

    total = 0
    started = time.monotonic()
    for i, episode in enumerate(todo, 1):
        count = await idx.ingest([episode])
        total += count
        elapsed = int(time.monotonic() - started)
        log(f"[{i}/{len(todo)}] +{count} windows ({total} total, {elapsed}s) "
            f"— {episode.title[:60]}")

    log(f"DONE: {total} windows from {len(todo)} episode(s)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
