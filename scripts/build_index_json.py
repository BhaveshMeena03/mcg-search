"""Write data/episode_index.json — the episode list the page needs.

    .venv/bin/python scripts/build_index_json.py

data/episodes.json is 19MB of transcript and stays out of git: it is
re-fetchable from YouTube and a deploy has no use for it, because the
searchable copy already lives in Pinecone.

What the page DOES need is the list it renders under "What's indexed" —
title, date, url, runtime. That is 85KB for all 406 episodes, small
enough to commit, which is what makes the app deployable from a clean
checkout.

Re-run after fetching new episodes.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FULL = ROOT / "data" / "episodes.json"
SLIM = ROOT / "data" / "episode_index.json"


def build(episodes: list[dict]) -> list[dict]:
    rows = [
        {
            "id": e["episode_id"],
            "title": e["title"],
            "url": e["url"],
            "published_at": e.get("published_at"),
            # The last segment's start time, which is close enough to a
            # runtime and costs nothing to compute.
            "seconds": int(e["segments"][-1]["t"]) if e.get("segments") else 0,
        }
        for e in episodes
    ]
    rows.sort(key=lambda r: r["published_at"] or "", reverse=True)
    return rows


def main() -> int:
    if not FULL.exists():
        print(f"{FULL} is missing — run scripts/fetch_episodes.py first.")
        return 1
    rows = build(json.loads(FULL.read_text()))
    SLIM.write_text(json.dumps(rows, ensure_ascii=False))
    hours = sum(r["seconds"] for r in rows) / 3600
    print(f"wrote {SLIM.name}: {len(rows)} episodes, {hours:.0f} hours, "
          f"{SLIM.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
