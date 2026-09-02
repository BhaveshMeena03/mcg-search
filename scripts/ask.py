"""Ask the MCG index a question.

    .venv/bin/python scripts/ask.py "what does ClawPump do"
    .venv/bin/python scripts/ask.py --hits "what does ClawPump do"

--hits prints the retrieved passages as well as the answer, which is what
you want when judging quality: a bad answer from good passages is a prompt
problem, a bad answer from bad passages is a retrieval problem, and you
cannot tell which from the answer alone.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.search import MCGIndex  # noqa: E402


async def main(argv: list[str]) -> int:
    show_hits = "--hits" in argv
    query = " ".join(a for a in argv if a != "--hits").strip()
    if not query:
        print(__doc__)
        return 1

    idx = MCGIndex()
    result = await idx.search(query)

    print(f"\nQ: {query}\n")
    print(result.answer)
    print()

    if show_hits:
        print("-" * 70)
        for h in result.hits:
            print(f"\n[{h.score:.3f}] {h.timestamp}  {h.title[:60]}")
            print(f"  {h.deep_link}")
            print(f"  {h.text[:280]}...")
        print()
    else:
        for h in result.hits:
            print(f"  [{h.score:.3f}] {h.timestamp}  {h.title[:60]}")

    print(f"\nmodel: {result.model}  hits: {len(result.hits)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
