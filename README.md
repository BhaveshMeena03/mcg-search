# MCG Search

Semantic search over the [MCG podcast](https://www.youtube.com/@MCG_live) —
**406 episodes, 275 hours**. Ask a question in plain language, get an answer
grounded in the transcripts, and press a timestamp to hear them say it.

Every MCG episode is one interview with one project, which suits this kind of
index unusually well: a question about a project has its answer concentrated
in one episode rather than scattered across a long multi-topic broadcast.

## How it works

    YouTube auto-captions  ->  overlapping windows  ->  voyage-3.5
      ->  Pinecone  ->  rerank-2.5-lite  ->  claude-haiku-4-5

No audio download and no transcription step — the whole channel publishes
auto-captions, so `yt-dlp` fetches the VTT directly and `app/captions.py`
strips YouTube's rolling-window overlap.

**The detail worth knowing before changing anything.** What gets embedded is
the plain passage. The per-line timestamps are rebuilt at answer time from a
compact list of line start times (`stamp()` in `app/search.py`), and handed
only to the model. Keeping timestamps out of the embedded text is what makes
the ranking depend on speech alone; giving them to the model is what lets a
citation land on the line actually used rather than wherever the passage
happened to begin.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    cp .env.example .env      # fill in the keys

`yt-dlp` needs a JS runtime on PATH (`brew install deno`) to get past
YouTube's challenge on caption downloads.

## Use

    # the site
    .venv/bin/uvicorn app.main:app --port 8110

    # ask from the terminal
    .venv/bin/python scripts/ask.py --hits "what does ClawPump do"

    # add whatever is new on the channel
    .venv/bin/python scripts/sync_new.py

`sync_new.py` is safe on a schedule: it lists the channel, fetches only what
is missing, indexes it and refreshes the page's episode list. Everything is
idempotent — vector ids are a hash of (episode, window start), so a re-run
overwrites rather than duplicates.

It has to run somewhere with a residential IP. From a datacenter address
YouTube refuses the video download with a bot check, though the channel
*listing* still works — which is why `deploy/com.mcgsearch.sync.plist` runs
the real sync on a laptop and the GitHub Action only does the listing half
and shouts when something new appears.

## Testing

    .venv/bin/pytest -q          # 179, all offline
    .venv/bin/ruff check .

Nothing in `tests/` calls Voyage, Pinecone or Anthropic, so the suite runs
identically on a laptop and on CI with no keys.

Answer quality is measured separately, because it costs money and needs the
network:

    .venv/bin/python scripts/grade.py --questions data/questions.txt

`grade.py` asserts on answers rather than printing them for a human to nod
at. It checks the things that fail silently: a fabricated URL, a citation
pointing at a line no passage contains, a correct answer that opens with a
denial, an investment question that wasn't declined, an injection that was
obeyed. Around 900 questions across six sets currently pass at ~96%, and
every genuine failure found so far is fixed and covered by a test.

## Deliberately not built yet

- **A term index.** Exact rare tokens in front of the reranker. Auto-captions
  mangle unusual proper nouns and this archive is nothing but unusual proper
  nouns — Scopl gets transcribed "Scopple". Left out because it has not
  earned its place: retrieval scores were identical from 100 episodes to 406.
- **Speaker labels.** Captions are words, not speakers, so the prompt says
  "the host" or "the founder" rather than guessing. An interview is two
  voices, so this is tractable — and it matters, because "did the host say
  that or the founder" is most of what you want from an interview archive.

## Safety

The archive is 406 founders describing their own projects, most of which have
tokens. The prompt refuses investment advice, declines before reporting when
a question asks it to judge or rank, and never writes a URL — it is given no
video addresses, so any link it produced would be invented.

`scripts/ingest_episodes.py` refuses to run when `PINECONE_INDEX` or
`PINECONE_NAMESPACE` names something listed in `PROTECTED_INDEXES` /
`PROTECTED_NAMESPACES`. One Pinecone account can hold several unrelated
projects, and a wrong `.env` in a scheduled job nobody is watching is the
only way this repo could reach data that isn't its own. Both lists are empty
by default.

## Deploying

`render.yaml` is a Render blueprint. Three secrets go in the dashboard —
`VOYAGE_API_KEY`, `PINECONE_API_KEY`, `ANTHROPIC_BASE_URL` — and everything
else is set in the file. `ANTHROPIC_API_KEY` is deliberately absent: the
inference proxy authenticates on a token inside its own URL, so a real
Anthropic credential there would sit in an environment doing nothing except
waiting to leak.

The transcripts (19MB) stay out of git — they are re-fetchable and the
searchable copy lives in Pinecone. The 85KB `data/episode_index.json` is
committed, which is what lets the page list what it can search from a clean
checkout.
