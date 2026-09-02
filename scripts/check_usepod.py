"""Does usepod actually serve claude-haiku-4-5, and serve it the way this
code calls it?

    export ANTHROPIC_BASE_URL='https://api.usepod.ai/proxy/<token>'
    .venv/bin/python scripts/check_usepod.py

The pitch is the same model at $0.40/$2.00 against Anthropic's
$1.00/$5.00. "Same model" is the entire premise, so it is the thing to
verify rather than read about — a proxy that quietly routes to something
cheaper would look like a bargain and be a downgrade.

Six checks, in rough order of how much each one proves:

  1. plain messages.create      the documented path. Proves the token
                                works and the proxy relays anything.
  2. WHICH MODEL ANSWERED       response.model, not the model asked for.
                                If these differ, stop here.
  3. beta.messages.create       the path app/search.py actually uses.
                                usepod documents /v1/messages only, so
                                whether the beta path relays is unknown.
  4. streaming                  needed by the SSE path on the web UI.
  5. prompt caching             search.py caches a ~1,400-token system
                                prompt. Cache reads bill at a tenth of
                                input. A proxy that drops cache_control
                                still "works" while quietly erasing much
                                of the saving.
  6. throughput                 every marketplace row showed 0.0 TPS.
                                This measures output tokens/second on a
                                real call.

Exit code is 0 only if 1-3 pass, because those are what the app needs to
run at all. 4-6 are reported and do not fail the run.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

MODEL = os.environ.get("SEARCH_MODEL", "claude-haiku-4-5")

# Long enough to be worth caching (the real system prompt is ~1,400
# tokens). Anthropic will not cache below a minimum, so a short probe
# would report "no caching" for the wrong reason.
CACHE_PROBE = ("You are a test harness. " * 200).strip()


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def bad(msg: str) -> None:
    print(f"  FAIL  {msg}")


def note(msg: str) -> None:
    print(f"        {msg}")


def _from_env_file(name: str) -> str:
    """Read one key out of .env, so this behaves like the app does.

    The app loads .env through pydantic-settings; this script does not use
    Settings because Settings requires every key to be present and this
    needs to run with only an Anthropic one.
    """
    path = ROOT / ".env"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


async def main() -> int:
    base_url = (os.environ.get("ANTHROPIC_BASE_URL", "").strip()
                or _from_env_file("ANTHROPIC_BASE_URL"))
    api_key = (os.environ.get("ANTHROPIC_API_KEY", "").strip()
               or _from_env_file("ANTHROPIC_API_KEY"))

    if not base_url:
        print("ANTHROPIC_BASE_URL is not set.\n")
        print("  export ANTHROPIC_BASE_URL='https://api.usepod.ai/proxy/<token>'")
        print("\nRun it again with that set. Without it this would test "
              "Anthropic direct and prove nothing about usepod.")
        return 2

    print(f"base_url : {base_url[:38]}...{base_url[-6:]}")
    print(f"model    : {MODEL}")
    # usepod carries its token in the URL path, so the SDK's api_key may be
    # ignored entirely. Send the existing one; if the proxy authenticates on
    # the path, this is harmless, and if it forwards the key we want to know.
    print(f"api_key  : {'set' if api_key else 'NOT SET'} "
          f"(usepod authenticates on the URL path; this may be unused)\n")

    client = AsyncAnthropic(api_key=api_key or "unused",
                            base_url=base_url).with_options(timeout=60.0)
    failed = False

    # --- 1. the documented path --------------------------------------------
    print("1. plain messages.create")
    try:
        started = time.monotonic()
        r1 = await client.messages.create(
            model=MODEL, max_tokens=64,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        )
        elapsed = time.monotonic() - started
        text = "".join(b.text for b in r1.content if b.type == "text").strip()
        ok(f"relayed in {elapsed:.1f}s — {text[:40]!r}")
    except Exception as exc:                                   # noqa: BLE001
        bad(f"{type(exc).__name__}: {str(exc)[:220]}")
        note("Nothing else can pass if this does not. Check the token.")
        return 1

    # --- 2. which model actually answered ----------------------------------
    print("\n2. which model answered")
    served = getattr(r1, "model", "") or ""
    if served == MODEL:
        ok(f"response.model == {served!r} — same model as asked for")
    else:
        bad(f"asked for {MODEL!r}, got {served!r}")
        note("This is the premise of the whole idea. A different model")
        note("means the price comparison is not like-for-like.")
        failed = True

    # --- 3. the path app/search.py uses ------------------------------------
    print("\n3. beta.messages.create (what app/search.py calls)")
    try:
        r3 = await client.beta.messages.create(
            model=MODEL, max_tokens=64,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        )
        ok(f"beta path relayed — model={getattr(r3, 'model', '?')!r}")
    except Exception as exc:                                   # noqa: BLE001
        bad(f"{type(exc).__name__}: {str(exc)[:220]}")
        note("app/search.py would need to move to client.messages.create.")
        failed = True

    # --- 4. streaming -------------------------------------------------------
    print("\n4. streaming (beta.messages.stream)")
    try:
        chunks = 0
        async with client.beta.messages.stream(
            model=MODEL, max_tokens=96,
            messages=[{"role": "user", "content": "Count from 1 to 8."}],
        ) as stream:
            async for _ in stream.text_stream:
                chunks += 1
            final = await stream.get_final_message()
        if chunks > 1:
            ok(f"streamed in {chunks} deltas — model={final.model!r}")
        else:
            bad(f"only {chunks} delta(s) — looks buffered, not streamed")
            note("The web UI's SSE path would still work but feel dead.")
    except Exception as exc:                                   # noqa: BLE001
        bad(f"{type(exc).__name__}: {str(exc)[:220]}")

    # --- 5. prompt caching --------------------------------------------------
    print("\n5. prompt caching")
    try:
        req = dict(
            model=MODEL, max_tokens=32,
            system=[{"type": "text", "text": CACHE_PROBE,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        )
        first = await client.beta.messages.create(**req)
        second = await client.beta.messages.create(**req)   # same bytes = hit

        def usage(r):
            u = r.usage
            return (getattr(u, "cache_creation_input_tokens", 0) or 0,
                    getattr(u, "cache_read_input_tokens", 0) or 0,
                    getattr(u, "input_tokens", 0) or 0)

        c1, r1_, i1 = usage(first)
        c2, r2_, i2 = usage(second)
        note(f"call 1: created={c1} read={r1_} input={i1}")
        note(f"call 2: created={c2} read={r2_} input={i2}")
        if r2_ > 0:
            ok(f"cache hit on the second call ({r2_} tokens read at ~1/10 price)")
        elif c1 > 0:
            bad("cache was created but not read back — no saving in practice")
        else:
            bad("no cache tokens reported at all")
            note("cache_control is being dropped or not relayed. The calls")
            note("still work; they just bill every token at full input rate.")
    except Exception as exc:                                   # noqa: BLE001
        bad(f"{type(exc).__name__}: {str(exc)[:220]}")

    # --- 6. throughput ------------------------------------------------------
    print("\n6. throughput (the 0.0 TPS question)")
    try:
        started = time.monotonic()
        r6 = await client.beta.messages.create(
            model=MODEL, max_tokens=400,
            messages=[{"role": "user",
                       "content": "Write a 300-word description of a sunset."}],
        )
        elapsed = time.monotonic() - started
        out = getattr(r6.usage, "output_tokens", 0) or 0
        note(f"{out} output tokens in {elapsed:.1f}s")
        if elapsed > 0 and out:
            tps = out / elapsed
            (ok if tps >= 10 else bad)(f"{tps:.1f} output tokens/sec")
            if tps < 10:
                note("Slow enough to be felt on an X reply or a live search.")
    except Exception as exc:                                   # noqa: BLE001
        bad(f"{type(exc).__name__}: {str(exc)[:220]}")

    print("\n" + "-" * 60)
    if failed:
        print("NOT SAFE TO SWITCH — see the failures above.")
        return 1
    print("Core path works. Run scripts/eval.py with ANTHROPIC_BASE_URL set")
    print("and compare the answers against the Claude-direct run.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
