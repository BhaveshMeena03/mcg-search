"""Does usepod actually serve claude-haiku-4-5, and serve it the way this
code calls it?

    export ANTHROPIC_BASE_URL='https://api.usepod.ai/proxy/<token>'
    .venv/bin/python scripts/check_usepod.py

The pitch is the same model at $0.40/$2.00 against Anthropic's
$1.00/$5.00. "Same model" is the entire premise, so it is the thing to
verify rather than read about — a proxy that quietly routes to something
cheaper would look like a bargain and be a downgrade.

Seven checks, in rough order of how much each one proves:

  1. relays, and WHICH ROUTE    the documented path. Reads X-Pod-Route:
                                the marketplace price and the pricier
                                centralized fallback both return a good
                                answer, and only this header tells them
                                apart.
  2. WHICH MODEL ANSWERED       response.model, not the model asked for.
                                An alias resolving to its dated snapshot
                                is the same model; a different family is
                                a substitution. If it substitutes, stop.
  3. beta.messages.create       the path app/search.py actually uses.
                                The docs say the proxy mirrors the
                                upstream surface, which is a claim, not
                                a measurement.
  4. streaming                  needed by the SSE path on a web UI.
  5. prompt caching             search.py caches a long system prompt and
                                cache reads bill at a tenth of input. A
                                proxy that drops cache_control still
                                "works" while erasing much of the saving.
  6. throughput                 every marketplace row showed 0.0 TPS.
                                Measures output tokens/second for real.
  7. price ceiling              asks for the advertised $0.40/$2.00 rate
                                explicitly. Listed is not the same as
                                purchasable.

SECURITY: usepod's auth token lives in the URL PATH, so it rides inside
every URL the SDK builds and surfaces in exception text. Everything this
script prints goes through _redact() first, and the real Anthropic key is
never sent to the proxy -- their docs say it is ignored, so forwarding it
would hand a live credential to a third party for nothing.

Exit code is 0 only if 1-3 pass, because those are what the app needs to
run at all. 4-7 are reported and do not fail the run.
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

# The minimum cacheable prefix is model-dependent, between 512 and 4096
# tokens, and a prefix under the floor silently does not cache. A 1,413
# token probe reported "no caching at all" against Anthropic direct --
# a false negative that would have been blamed on the proxy. Sized well
# clear of the top of that range so a miss here means something real.
CACHE_PROBE = ("You are a test harness for an API proxy. " * 700).strip()


# Every string printed by this script goes through _redact() first.
#
# usepod puts the auth token in the URL PATH, not in a header, which makes
# it far easier to leak than a normal API key: it rides inside every URL
# the SDK builds, so it turns up in exception messages, connection errors
# and stack traces. An SDK error printed raw would put a live credential
# into terminal scrollback and CI logs. Redaction is centralised here so
# a future print cannot forget it.
_SECRETS: list[str] = []


def _redact(text: str) -> str:
    out = str(text)
    for secret in _SECRETS:
        if secret and len(secret) > 6:
            out = out.replace(secret, "<redacted>")
    return out


def _exc(exc: Exception) -> str:
    """An exception, safe to print.

    Redact BEFORE truncating. Truncating first can cut a token in half
    and print the first half, which redaction then no longer matches.
    """
    return f"{type(exc).__name__}: {_redact(exc)[:220]}"


def ok(msg: str) -> None:
    print(f"  PASS  {_redact(msg)}")


def bad(msg: str) -> None:
    print(f"  FAIL  {_redact(msg)}")


def note(msg: str) -> None:
    print(f"        {_redact(msg)}")


def _token_from(base_url: str) -> str:
    """The secret inside a proxy URL, or "" when there isn't one.

    usepod's token is the last path segment. Two things must not be
    registered as secrets: the literal "v1" that ends an OpenAI-style
    base URL, and a bare host with no path — redacting either would
    strip ordinary words out of every message this script prints.
    """
    last = base_url.rstrip("/").rsplit("/", 1)[-1]
    if not last or last == "v1" or "." in last:
        return ""
    return last


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

    # Register the token for redaction BEFORE anything can be printed.
    token = _token_from(base_url)
    if token:
        _SECRETS.append(token)
    if api_key:
        _SECRETS.append(api_key)

    # Show the endpoint, never the credential in it. The earlier version
    # of this line printed base_url[:38] + the last 6 characters, which
    # revealed most of a short token.
    print(f"base_url : {_redact(base_url)}")
    print(f"model    : {MODEL}")

    # NEVER send the real Anthropic key to a third-party proxy.
    #
    # usepod authenticates on the token in the URL path and its docs say
    # the SDK's api_key "is ignored - use any placeholder". Passing the
    # real sk-ant-... would hand a live Anthropic credential to another
    # company's servers for no benefit at all. Against api.anthropic.com
    # this placeholder is expected to fail on auth, which is correct:
    # this script is for testing a proxy, not Anthropic direct.
    proxied = "api.anthropic.com" not in base_url
    key = "unused-auth-is-in-the-url" if proxied else api_key
    print("api_key  : " + ("placeholder (auth is the token in the URL)"
                           if proxied else "real key (direct to Anthropic)")
          + "\n")

    client = AsyncAnthropic(api_key=key or "unused",
                            base_url=base_url).with_options(timeout=60.0)
    failed = False

    # --- 1. the documented path, and WHICH ROUTE served it ------------------
    print("1. plain messages.create")
    try:
        started = time.monotonic()
        # Raw response so the proxy's own headers are readable. X-Pod-Route
        # is the one that matters: the $0.40/$2.00 marketplace price and the
        # $0.80/$4.00 centralized fallback both return a perfectly good
        # answer, and only this header distinguishes them. A "pass" served
        # by fallback would still be cheaper than Anthropic, but it is not
        # the thing being evaluated.
        raw = await client.messages.with_raw_response.create(
            model=MODEL, max_tokens=64,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        )
        elapsed = time.monotonic() - started
        r1 = raw.parse()
        text = "".join(b.text for b in r1.content if b.type == "text").strip()
        ok(f"relayed in {elapsed:.1f}s — {text[:40]!r}")

        route = raw.headers.get("x-pod-route")
        balance = raw.headers.get("x-balance-remaining")
        if route:
            note(f"X-Pod-Route: {route}")
            if "marketplace" in route.lower():
                ok("served by the MARKETPLACE — the $0.40/$2.00 price")
            else:
                bad(f"served by {route!r}, not the marketplace")
                note("Works, but this is the centralized fallback tier")
                note("($0.80/$4.00), not the price the pitch is based on.")
        elif proxied:
            note("no X-Pod-Route header — cannot tell which route served it")
        if balance:
            note(f"X-Balance-Remaining: {balance}")
    except Exception as exc:                                   # noqa: BLE001
        bad(_exc(exc))
        note("Nothing else can pass if this does not. Check the token,")
        note("and that the balance is funded.")
        return 1

    # --- 2. which model actually answered ----------------------------------
    print("\n2. which model answered")
    served = getattr(r1, "model", "") or ""
    # An alias resolves to a dated snapshot: asking for claude-haiku-4-5
    # and being served claude-haiku-4-5-20251001 is the same model, and
    # is what Anthropic direct does too. Only a different family is a
    # substitution, so match on the prefix rather than on equality.
    if served == MODEL:
        ok(f"response.model == {served!r}")
    elif served.startswith(MODEL):
        ok(f"{served!r} — the dated snapshot of {MODEL!r}, same model")
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
        bad(_exc(exc))
        note("app/search.py would need to move to client.messages.create.")
        failed = True

    # --- 4. streaming -------------------------------------------------------
    print("\n4. streaming (beta.messages.stream)")
    try:
        chunks = 0
        # Ask for enough output that a genuinely streamed response must
        # arrive in many deltas. "Count to 8" fits in one or two, so a
        # buffered proxy and a streaming one look identical.
        async with client.beta.messages.stream(
            model=MODEL, max_tokens=400,
            messages=[{"role": "user",
                       "content": "Write 200 words about the sea."}],
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
        bad(_exc(exc))

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
        bad(_exc(exc))

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
        bad(_exc(exc))

    # --- 7. can the advertised price actually be bought? --------------------
    print("\n7. price ceiling at the advertised marketplace rate")
    if not proxied:
        note("skipped — only meaningful through the proxy")
    else:
        try:
            # Prices are USDC microunits per million tokens; 1 USDC = 1e6.
            # The marketplace listing for claude-haiku-4-5 is $0.40 in /
            # $2.00 out, so these ceilings are exactly the advertised
            # price. If no route can serve at or below them the request
            # is REJECTED rather than silently billed higher -- which is
            # the honest failure, and the one worth knowing about before
            # switching a live surface over.
            r7 = await client.messages.with_raw_response.create(
                model=MODEL, max_tokens=32,
                messages=[{"role": "user", "content": "Reply with: pong"}],
                extra_headers={
                    "X-Pod-Max-Price-Input": "400000",
                    "X-Pod-Max-Price-Output": "2000000",
                },
            )
            r7.parse()
            ok("a route served at or below $0.40/$2.00 per M")
            if r7.headers.get("x-pod-route"):
                note(f"X-Pod-Route: {r7.headers.get('x-pod-route')}")
        except Exception as exc:                              # noqa: BLE001
            bad(_exc(exc))
            note("No route can serve at the advertised price. The model is")
            note("listed at $0.40/$2.00 but is not purchasable there.")

    print("\n" + "-" * 60)
    if failed:
        print("NOT SAFE TO SWITCH — see the failures above.")
        return 1
    print("Core path works. Run scripts/eval.py with ANTHROPIC_BASE_URL set")
    print("and compare the answers against the Claude-direct run.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
