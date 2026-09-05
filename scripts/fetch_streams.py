"""Fetch MCG's live streams, which the /videos tab does not list.

    .venv/bin/python scripts/fetch_streams.py --latest 10
    .venv/bin/python scripts/fetch_streams.py --all

The channel has 221 of these and none were in the archive, because
YouTube keeps live streams on a separate tab and scripts/fetch_episodes.py
lists /videos. That was a quarter of the content missing — and the
better-watched quarter: the streams run 96-408 views against about 60
for an interview clip.

They are four-hour broadcasts, market commentary with founder interviews
inside. The interviews are cut out and uploaded separately a day or two
later, so roughly 43% of a stream is already in the archive as a clip.
The ingest drops those windows; see app/overlap.py. What is left is the
market talk, which exists nowhere else.

Written to data/episodes.json alongside the clips, with format="stream".
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.episode_store import merge  # noqa: E402
from scripts.fetch_episodes import (  # noqa: E402
    SLEEP_BETWEEN,
    _cookie_args,
    _env,
    fetch,
)

OUT = ROOT / "data" / "episodes.json"
STREAMS_URL = "https://www.youtube.com/@MCG_live/streams"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stream_ids(limit: int | None, cookies: str | None) -> list[str]:
    import subprocess

    from scripts.fetch_episodes import YTDLP
    cmd = [YTDLP, "--flat-playlist", "--print", "%(id)s",
           "--remote-components", "ejs:github", *_cookie_args(cookies)]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(STREAMS_URL)
    out = subprocess.run(cmd, capture_output=True, text=True,
                         check=False, env=_env())
    ids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not ids:
        err = (out.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        log(f"stream listing failed — {err[:140]}")
    return ids


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cookies", default=None)
    args = ap.parse_args(argv)
    if not args.latest and not args.all:
        print(__doc__)
        return 1

    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    have = {e["episode_id"] for e in existing}

    ids = stream_ids(None if args.all else args.latest, args.cookies)
    if not ids:
        return 1
    todo = [v for v in ids if v not in have]
    log(f"{len(ids)} streams listed, {len(have)} already on file, "
        f"{len(todo)} to fetch")

    fetched = 0
    for i, vid in enumerate(todo):
        ep = fetch(vid, args.cookies)
        if ep:
            # The one thing this script adds over fetch_episodes: the
            # format, which decides whether the ingest deduplicates it.
            ep["format"] = "stream"
            merge([ep], OUT)          # one at a time; a long run keeps its work
            fetched += 1
        if i < len(todo) - 1:
            time.sleep(SLEEP_BETWEEN)

    total = len(json.loads(OUT.read_text())) if OUT.exists() else 0
    log(f"fetched {fetched} streams; {total} items on file")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
