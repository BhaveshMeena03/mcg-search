"""Check YouTube for new MCG content, fetch and index whatever is new.

    .venv/bin/python scripts/sync_new.py
    .venv/bin/python scripts/sync_new.py --check      # report only
    .venv/bin/python scripts/sync_new.py --limit 30   # newest N ids only

BOTH TABS, STREAMS FIRST. Measured across 50 streams and the 68
interviews published inside their window: 96% of interviews are mostly
contained in a stream, 0% partially, 4% not at all. Perfectly bimodal,
which is what you see when the clips are literally cut out of the
broadcast rather than re-recorded.

So the stream is the source and lands first — same day — while the clips
appear a day or two later. Watching streams means full coverage within
hours of broadcast instead of waiting. The clips are still fetched and
still worth having: a 42-minute episode titled for its project is a much
better retrieval target, and a much better place to send a reader, than
the same conversation two hours into a four-hour market update. The
ingest drops the stream windows the clips already cover.

Safe to run on a schedule. Everything it does is idempotent: episodes
already on file are skipped, and the vector ids are a hash of
(episode, window start) so re-indexing overwrites rather than
duplicates.

WHERE THIS CAN RUN. On a laptop, anywhere. Not on CI without a
residential proxy: from a datacenter IP YouTube answers "Sign in to
confirm you're not a bot" for the video, though the channel LISTING
still works — that asymmetry is measured, not assumed, and it is why
--check is useful on CI while a full sync is not.

MCG streams Tuesday and Thursday at 12:30 EST, so daily is enough.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.overlap import nearby, reference_shingles  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.search import MCGIndex  # noqa: E402
from scripts.build_index_json import build as build_index  # noqa: E402
from scripts.fetch_episodes import SLEEP_BETWEEN, channel_ids, fetch  # noqa: E402
from scripts.fetch_streams import stream_ids  # noqa: E402
from scripts.ingest_episodes import check_target, ensure_index  # noqa: E402

DATA = ROOT / "data" / "episodes.json"
SLIM = ROOT / "data" / "episode_index.json"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def on_file() -> tuple[list[dict], set[str]]:
    if not DATA.exists():
        return [], set()
    eps = json.loads(DATA.read_text())
    return eps, {e["episode_id"] for e in eps}


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what is new and exit without fetching")
    ap.add_argument("--limit", type=int, default=40,
                    help="how far back down the channel to look")
    ap.add_argument("--cookies", default=None)
    args = ap.parse_args(argv)

    settings = get_settings()
    # Same guard as the ingest script. A scheduled job writing to the
    # wrong index is worse than a manual one, because nobody is watching.
    check_target(settings)

    episodes, have = on_file()

    # Streams first: they are the broadcast the clips come out of, and
    # they land the same day.
    stream_new = [v for v in stream_ids(args.limit, args.cookies)
                  if v not in have]
    clip_ids = channel_ids(args.limit, args.cookies)
    if not clip_ids and not stream_new:
        log("both listings failed — nothing to do")
        return 1
    clip_new = [v for v in clip_ids if v not in have]

    # Format matters at ingest: a stream gets deduplicated against the
    # clips near it, a clip never does.
    new = [(v, "stream") for v in stream_new] + \
          [(v, "interview") for v in clip_new]

    log(f"{len(have)} on file — {len(stream_new)} new stream(s), "
        f"{len(clip_new)} new clip(s)")
    if not new:
        return 0
    if args.check:
        log("new: " + ", ".join(f"{v} ({k})" for v, k in new))
        # Non-zero so a scheduler or CI job can treat "there is work" as
        # a signal without needing to parse the output.
        return 2

    fetched = []
    for i, (vid, kind) in enumerate(new):
        ep = fetch(vid, args.cookies)
        if ep:
            ep["format"] = kind
            fetched.append(ep)
        if i < len(new) - 1:
            time.sleep(SLEEP_BETWEEN)

    if not fetched:
        log("nothing fetched — likely a bot check; run this from a laptop")
        return 1

    # Write transcripts first. If indexing dies the fetch is not lost.
    from app.episode_store import merge
    merge(fetched, DATA)

    ensure_index(settings)
    idx = MCGIndex()
    all_eps, _ = on_file()
    clips = [e for e in all_eps if e.get("format", "interview") == "interview"]
    total = 0
    # Clips before streams, so a stream is deduplicated against the clips
    # from the same broadcast even when both arrive in one run.
    for ep in sorted(fetched, key=lambda e: e.get("format") == "stream"):
        reference = None
        if ep.get("format") == "stream":
            reference = reference_shingles(nearby(clips, ep.get("published_at")))
        total += await idx.ingest([Episode(**ep)], reference)
        log(f"indexed {ep.get('format', 'interview'):9} {ep['title'][:52]}")

    # The page reads this, and a deploy has no transcripts.
    all_eps, _ = on_file()
    SLIM.write_text(json.dumps(build_index(all_eps), ensure_ascii=False))

    log(f"done: {len(fetched)} episodes, {total} windows, "
        f"{len(all_eps)} on file")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
