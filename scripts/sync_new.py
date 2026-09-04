"""Check YouTube for new MCG episodes, fetch and index whatever is new.

    .venv/bin/python scripts/sync_new.py
    .venv/bin/python scripts/sync_new.py --check      # report only
    .venv/bin/python scripts/sync_new.py --limit 30   # newest N ids only

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
from app.schemas import Episode  # noqa: E402
from app.search import MCGIndex  # noqa: E402
from scripts.build_index_json import build as build_index  # noqa: E402
from scripts.fetch_episodes import SLEEP_BETWEEN, channel_ids, fetch  # noqa: E402
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
    ids = channel_ids(args.limit, args.cookies)
    if not ids:
        log("channel listing failed — nothing to do")
        return 1
    new = [v for v in ids if v not in have]

    log(f"channel has {len(ids)} recent, {len(have)} on file, {len(new)} new")
    if not new:
        return 0
    if args.check:
        log(f"new: {', '.join(new)}")
        # Non-zero so a scheduler or CI job can treat "there is work" as
        # a signal without needing to parse the output.
        return 2

    fetched = []
    for i, vid in enumerate(new):
        ep = fetch(vid, args.cookies)
        if ep:
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
    total = 0
    for ep in fetched:
        total += await idx.ingest([Episode(**ep)])
        log(f"indexed {ep['title'][:60]}")

    # The page reads this, and a deploy has no transcripts.
    all_eps, _ = on_file()
    SLIM.write_text(json.dumps(build_index(all_eps), ensure_ascii=False))

    log(f"done: {len(fetched)} episodes, {total} windows, "
        f"{len(all_eps)} on file")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
