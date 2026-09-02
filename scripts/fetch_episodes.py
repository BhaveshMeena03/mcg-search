"""Pull MCG captions from YouTube into the Episode JSON the index ingests.

Uses yt-dlp to download auto-generated VTT subtitles. No video, no audio,
no transcription step — the whole channel is YouTube uploads with
auto-captions on, so the transcript is already written and this just
fetches it.

    .venv/bin/python scripts/fetch_episodes.py --latest 15
    .venv/bin/python scripts/fetch_episodes.py --all
    .venv/bin/python scripts/fetch_episodes.py VIDEO_ID [VIDEO_ID ...]

Resumable. Anything already in data/episodes.json is skipped, so an
interrupted run costs only what is missing — which matters at 407
episodes, where this is a job you start, stop, and come back to.

If YouTube starts refusing (bot check / 429), pass browser cookies:

    .venv/bin/python scripts/fetch_episodes.py --cookies chrome --all
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.captions import coalesce, parse_vtt  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.episode_store import merge as merge_episodes  # noqa: E402

OUT = ROOT / "data" / "episodes.json"

# Be polite. YouTube throttles rapid pulls, and at 407 episodes this is the
# difference between a slow job and a blocked one. It is also the dominant
# cost of a full fetch: 407 x 12s is about 80 minutes of pure waiting, so
# lower it at your own risk and watch for 429s if you do.
SLEEP_BETWEEN = 12


def _ytdlp() -> str:
    """Prefer the project venv, fall back to PATH."""
    local = ROOT / ".venv" / "bin" / "yt-dlp"
    return str(local) if local.exists() else (shutil.which("yt-dlp") or "yt-dlp")


YTDLP = _ytdlp()

# yt-dlp's failure modes look the same from outside but need opposite
# responses: a blocked IP needs cookies or a different host, a video with
# no captions yet just needs waiting. Naming which one turns a silent
# stall into something actionable.
_BLOCK_SIGNS = (
    "sign in to confirm", "not a bot", "confirm you're not a bot",
    "http error 403", "http error 429", "too many requests",
    "unable to extract", "player response", "failed to extract",
)


def _env() -> dict[str, str]:
    """yt-dlp's environment, with homebrew's bin on PATH.

    --remote-components needs a JS runtime (deno) to solve YouTube's
    challenge, and a GUI-launched process does not inherit a shell PATH
    that can find it.
    """
    return {**os.environ,
            "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}


def _cookie_args(cookies: str | None) -> list[str]:
    return ["--cookies-from-browser", cookies] if cookies else []


def _diagnose(stderr: str) -> str:
    err = (stderr or "").lower()
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    if any(sign in err for sign in _BLOCK_SIGNS):
        return f"BLOCKED by YouTube (bot check / rate limit) — {lines[-1][:160]}"
    if "no subtitles" in err or "no automatic captions" in err:
        return "no captions published for this video"
    if not lines:
        # yt-dlp exits quietly when a video simply HAS no captions: it
        # writes no file and reports no error, so the old "(no stderr)"
        # was indistinguishable from a real fault. One episode in 407 is
        # like this (eqq_oEyDusU, "Reid Moncada from Fitted") — public,
        # 34 minutes, never auto-captioned by YouTube. Naming it is the
        # difference between a mystery to re-investigate and a known,
        # permanent gap.
        return ("no captions available (yt-dlp exited quietly) — verify "
                "with: yt-dlp --list-subs <url>")
    return lines[-1][:160]


def fetch(video_id: str, cookies: str | None = None,
          attempts: int = 3) -> dict | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    # YouTube gates caption downloads behind a JS challenge.
    # --remote-components pulls yt-dlp's solver.
    common = ["--remote-components", "ejs:github", *_cookie_args(cookies)]
    env = _env()
    proc = None
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # One invocation downloads captions AND prints title + date.
            # Two calls per episode trips the rate limit; --print alone
            # implies simulate, so --no-simulate is what makes it write.
            proc = subprocess.run(
                [YTDLP, "--skip-download", "--write-auto-sub", "--write-sub",
                 "--sub-lang", "en", "--sub-format", "vtt", *common,
                 "--print", "%(title)s\t%(upload_date)s", "--no-simulate",
                 "-o", str(tmp_path / "%(id)s.%(ext)s"), url],
                capture_output=True, text=True, check=False, env=env,
            )
            first = (proc.stdout.strip().splitlines() or [video_id])[0]
            title, _, raw_date = first.partition("\t")
            title = title or video_id
            # yt-dlp gives YYYYMMDD, or "NA" when it has no date. The date
            # is what lets an answer say how current a claim is.
            published_at = (f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                            if len(raw_date) == 8 and raw_date.isdigit()
                            else None)
            vtts = list(tmp_path.glob("*.vtt"))
            if vtts:
                segments = coalesce(parse_vtt(
                    vtts[0].read_text(encoding="utf-8", errors="ignore")
                ))
                if segments:
                    print(f"  ok {video_id}: {len(segments)} segments — {title[:60]}")
                    return {
                        "episode_id": video_id, "title": title, "url": url,
                        "platform": "youtube", "published_at": published_at,
                        "segments": segments,
                    }
        why = _diagnose(proc.stderr)
        if attempt < attempts:
            wait = 20 * attempt
            print(f"  .. {video_id} failed (try {attempt}/{attempts}): {why} "
                  f"— waiting {wait}s")
            time.sleep(wait)
    print(f"  !! {video_id} failed after {attempts} tries: "
          f"{_diagnose(proc.stderr if proc else '')}")
    return None


def channel_ids(limit: int | None = None, cookies: str | None = None) -> list[str]:
    """Video ids from the channel, newest first.

    Deliberately NOT the same request as the per-video fetch: listing the
    channel is one cheap call and is not what the bot check guards.
    """
    cmd = [YTDLP, "--flat-playlist", "--print", "%(id)s",
           "--remote-components", "ejs:github", *_cookie_args(cookies)]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(get_settings().youtube_channel)
    out = subprocess.run(cmd, capture_output=True, text=True,
                         check=False, env=_env())
    ids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not ids:
        err = (out.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"  !! channel listing failed — {err[:140]}")
    return ids[:limit] if limit else ids


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    cookies = None
    if argv[0] == "--cookies":
        cookies, argv = argv[1], argv[2:]

    if argv and argv[0] == "--latest":
        ids = channel_ids(int(argv[1]) if len(argv) > 1 else 10, cookies)
    elif argv and argv[0] == "--all":
        ids = channel_ids(None, cookies)
        print(f"Channel lists {len(ids)} videos.")
    else:
        ids = argv

    # Resume: keep what a previous run already fetched.
    existing: list[dict] = []
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            existing = []
    done = {e["episode_id"] for e in existing}
    todo = [v for v in ids if v not in done]

    print(f"Fetching {len(todo)} episode(s) ({len(done)} already cached)...")
    fetched = 0
    for i, vid in enumerate(todo):
        ep = fetch(vid, cookies)
        if ep:
            # Save one at a time. A run that spans hours must never write
            # back the stale snapshot it started from.
            merge_episodes([ep], OUT)
            fetched += 1
        if i < len(todo) - 1:
            time.sleep(SLEEP_BETWEEN)

    total = len(json.loads(OUT.read_text())) if OUT.exists() else 0
    print(f"\nFetched {fetched} new; {total} episodes on file -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
