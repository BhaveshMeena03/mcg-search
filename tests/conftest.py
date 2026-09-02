"""Test setup.

Every test here runs offline. Nothing in this suite calls Voyage, Pinecone
or Anthropic — those are covered by scripts/check_usepod.py and
scripts/eval.py, which cost money and need network. The point of this
suite is the logic that decides what gets indexed and what a citation
points at, which is pure and should be tested as such.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Settings requires these to construct. Set before any app import so a
# machine with no .env still runs the suite.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX", "mcg-search-test")
os.environ.setdefault("PINECONE_NAMESPACE", "mcg-test")
