# MCG search

Semantic search over the [MCG podcast](https://www.youtube.com/@MCG_live) —
407 episodes, 278 hours. Ask a question, get an answer grounded in the
transcripts with citations that deep-link to the exact second in the video.

Every MCG episode is one interview with one project, which suits this kind
of index well: a question about a project has its answer concentrated in
one episode rather than scattered across a long multi-topic broadcast.

## How it works

    YouTube auto-captions  ->  overlapping windows  ->  voyage-3.5
      ->  Pinecone  ->  rerank-2.5-lite  ->  claude-haiku-4-5

No audio download and no transcription step. The whole channel publishes
auto-captions, so `yt-dlp` fetches the VTT directly and `app/captions.py`
strips YouTube's rolling-window overlap.

The one detail worth knowing before changing anything: each window is
stored **twice**, as `text` (embedded, and what a reader sees) and
`text_ts` (the same lines with a timestamp on each, given only to the
model). That split is what lets a citation land on the line that was
actually used instead of on wherever the passage happened to begin. See
the `_windows` docstring in `app/search.py`.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    cp .env.example .env      # fill in the keys

`yt-dlp` needs a JS runtime on PATH (`brew install deno`) to get past
YouTube's challenge on caption downloads.

## Use

    .venv/bin/python scripts/fetch_episodes.py --latest 15
    .venv/bin/python scripts/ingest_episodes.py
    .venv/bin/python scripts/ask.py --hits "what does ClawPump do"

Both scripts resume. `fetch` skips episodes already on file; `ingest` asks
Pinecone which episodes are already in and only does the rest. A full
fetch is dominated by the 12-second politeness gap between videos — about
80 minutes of waiting for 407 episodes — so it is a job you start and come
back to, not one you sit through.

## What this shares with the Market Bubble index, and what it doesn't

Ported from [bullpen-ai](https://github.com/BhaveshMeena03/bullpen-ai),
which does the same thing for the Market Bubble podcast. The parts that
took measuring came across as-is: the caption overlap stripping, the
window split above, and the prompt rules — most of which were written
after a specific wrong answer went out in public.

Deliberately **not** ported yet:

- **The term index** (`terms.py` / `names.py` over there). It puts
  passages containing a rare exact token in front of the reranker. I
  expect this to be the first thing this index needs, because auto-captions
  mangle unusual proper nouns and this archive is nothing but unusual
  proper nouns — a project called Scopl gets transcribed as "Scopple". I
  left it out so I can measure whether it's needed rather than assume it.
- **Speaker labels.** Neither index knows who is talking; captions are
  words, not speakers. It should be easier here than on Market Bubble
  (an interview is two voices, a co-hosted broadcast with four guests is
  not), and it matters more, because "did the host say that or the
  founder" is most of what you'd want to know from an interview archive.
- **The deep + shallow rerank union.** A measured quality win on the
  other archive. Worth trying if retrieval comes up short.

## Tests

    .venv/bin/pytest -q
    .venv/bin/ruff check .

36 tests, all offline — nothing in `tests/` calls Voyage, Pinecone or
Anthropic, so the suite runs the same on a laptop and on CI with no keys.
They cover the four things that fail silently rather than loudly: caption
overlap stripping, the embedded-vs-timestamped window invariant, the
concurrent-write behaviour of `episodes.json`, and the refusal to write
anywhere near the live Bullpen index.

Anything needing a real key lives in `scripts/` instead — `eval.py` for
answer quality, `check_usepod.py` for an alternative inference provider.
Those cost money to run, which is why they are not tests.

## Not done

No web UI and no API server yet. This is the pipeline plus a CLI to judge
whether the answers are good enough to build the rest on.

## Safety

`scripts/ingest_episodes.py` refuses to run if `PINECONE_INDEX` or
`PINECONE_NAMESPACE` points at the live Bullpen index. A wrong `.env` is
the only way this repo could affect a running product, so it's a hard stop
rather than a note in a readme.
