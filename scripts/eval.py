"""Run a list of questions and print the answers for eyeballing.

    .venv/bin/python scripts/eval.py
    .venv/bin/python scripts/eval.py data/questions.txt

No automatic grading. Judging whether an answer is right needs someone who
knows the archive, and a scoring script that guesses would just launder a
guess into a number. What this does is make the same questions easy to
re-run after a change, so a regression is visible.

Questions live one per line in data/questions.txt. A line starting with #
is a comment, and a blank line is ignored.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.search import MCGIndex  # noqa: E402

DEFAULT = ROOT / "data" / "questions.txt"


async def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT
    if not path.exists():
        print(f"no question file at {path}")
        return 1
    questions = [ln.strip() for ln in path.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]

    idx = MCGIndex()
    for i, q in enumerate(questions, 1):
        result = await idx.search(q)
        top = result.hits[0] if result.hits else None
        print(f"\n{'=' * 72}")
        print(f"[{i}/{len(questions)}] {q}")
        print(f"{'=' * 72}")
        print(result.answer)
        if top:
            print(f"\n  top hit: [{top.score:.3f}] {top.timestamp} "
                  f"{top.title[:55]}")
    print(f"\n{len(questions)} question(s) run.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
