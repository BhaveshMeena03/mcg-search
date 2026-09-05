"""Write a TL;DR and timestamped topics for every episode.

    .venv/bin/python scripts/summarize.py
    .venv/bin/python scripts/summarize.py --limit 5      # try a few first
    .venv/bin/python scripts/summarize.py --only VIDEO_ID
    .venv/bin/python scripts/summarize.py --force        # redo existing

Resumable, and saves after every episode. 406 episodes is roughly half an
hour and about two dollars, which is long enough that a crash halfway
through must not cost the half that worked.

Output is data/summaries.json, which IS committed: the page needs it and
it is small, where the transcripts it was built from are 19MB and stay
out of git.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import Episode  # noqa: E402
from app.search import MCGIndex  # noqa: E402
from app.summaries import (  # noqa: E402
    SUMMARY_PROMPT,
    looks_misattributed,
    parse,
    stamped_transcript,
    validate,
)

DATA = ROOT / "data" / "episodes.json"
OUT = ROOT / "data" / "summaries.json"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_existing() -> dict:
    if not OUT.exists():
        return {}
    try:
        return json.loads(OUT.read_text())
    except json.JSONDecodeError:
        # Never silently start over: half an hour of work is in here.
        raise SystemExit(
            f"{OUT} is not valid JSON — refusing to overwrite"
        ) from None


def save(summaries: dict) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summaries, ensure_ascii=False, indent=1))
    tmp.replace(OUT)          # atomic, so an interrupt cannot truncate it


async def summarize_one(idx: MCGIndex, episode: Episode,
                        args_pin: bool = False) -> dict | None:
    client = idx._anthropic.with_options(
        timeout=idx._settings.search_timeout_seconds
    )
    request = {
        "model": idx._settings.search_model,
        # Long enough for ten topics and a TL;DR with room to spare.
        "max_tokens": 2000,
        "system": [{"type": "text", "text": SUMMARY_PROMPT,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text",
             "text": f"Episode title: {episode.title}\n\n"
                     f"{stamped_transcript(episode)}"},
        ]}],
        # Deliberately NOT pinned to a provider.
        #
        # A pin is a hard constraint: when the pinned provider cannot
        # serve, the request 503s rather than falling back. That is the
        # right trade for a page somebody is waiting on, where a
        # predictable 4s beats a lottery between 6 and 40. It is the
        # wrong trade here. This is a batch job with nobody watching, so
        # it wants whatever provider is up, however slow — and the first
        # run of it died on "All upstream providers temporarily
        # unavailable" while a pin was in force.
        **({} if not args_pin else idx._proxy_headers_kwargs()),
    }
    response = await client.beta.messages.create(**request)
    if response.stop_reason == "refusal":
        log(f"  refused: {episode.title[:50]}")
        return None
    raw = "".join(b.text for b in response.content if b.type == "text")
    tldr, topics = parse(raw)
    topics = validate(topics, episode)
    if not tldr or not topics:
        log(f"  unusable output: {episode.title[:50]}")
        return None
    if looks_misattributed(tldr):
        # Rather than ship a TL;DR whose first word is a wrong name.
        log(f"  misattributed opening, skipping: {episode.title[:44]}")
        return None
    return {"tldr": tldr, "topics": topics, "model": response.model}


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pin", action="store_true",
                    help="pin the upstream provider; off by default because "
                         "a batch job wants availability over latency")
    args = ap.parse_args(argv)

    if not DATA.exists():
        log("data/episodes.json is missing — run sync_new.py first.")
        return 1
    episodes = [Episode(**e) for e in json.loads(DATA.read_text())]
    summaries = load_existing()

    todo = episodes
    if args.only:
        todo = [e for e in todo if e.episode_id == args.only]
    elif not args.force:
        todo = [e for e in todo if e.episode_id not in summaries]
    if args.limit:
        todo = todo[:args.limit]

    log(f"{len(summaries)} already summarised, {len(todo)} to do")
    if not todo:
        return 0

    idx = MCGIndex()
    done = failed = 0
    started = time.monotonic()
    for i, episode in enumerate(todo, 1):
        try:
            result = await summarize_one(idx, episode, args.pin)
        except Exception as exc:                              # noqa: BLE001
            log(f"[{i}/{len(todo)}] FAILED {type(exc).__name__}: "
                f"{str(exc)[:90]} — {episode.title[:40]}")
            failed += 1
            continue
        if not result:
            failed += 1
            continue
        summaries[episode.episode_id] = result
        save(summaries)          # after every one; a crash costs one episode
        done += 1
        rate = (time.monotonic() - started) / i
        log(f"[{i}/{len(todo)}] {len(result['topics'])} topics, "
            f"{rate:.1f}s/ep — {episode.title[:52]}")

    log(f"DONE: {done} written, {failed} skipped, {len(summaries)} on file")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
