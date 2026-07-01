"""Knowledge-base retrieval: BM25 over the markdown sections in ``kb/``.

Each ``## Section`` of each document is one chunk with a stable id
``<file-stem>#<section-slug>`` that the agent must cite as [doc:<id>].
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..data import ROOT
from . import Tool

KB = ROOT / "kb"
STOP = set(
    "the a an and or of to in on for by with is are be as at from that this it its which their than into "
    "not no can must should was were has have had over under each per".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, stop words removed, truncated to 6 characters: a crude
    but effective stemmer for a small corpus (plausible / plausibility -> plausi)."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:6] for t in toks if t not in STOP]


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@lru_cache(maxsize=1)
def load_chunks(kb: Path = KB) -> tuple[Chunk, ...]:
    out = []
    for p in sorted(kb.glob("*.md")):
        doc_title, section, buf = p.stem, None, []
        for line in p.read_text().splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                if section:
                    out.append(Chunk(f"{p.stem}#{_slug(section)}", f"{doc_title}: {section}", " ".join(buf).strip()))
                section, buf = line[3:].strip(), []
            elif line.strip():
                buf.append(line.strip())
        if section:
            out.append(Chunk(f"{p.stem}#{_slug(section)}", f"{doc_title}: {section}", " ".join(buf).strip()))
    return tuple(out)


class BM25:
    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(c.title + " " + c.title + " " + c.text) for c in chunks]  # titles count twice
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = sum(
                self.idf.get(t, 0.0)
                * tf[t]
                * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for t in q
                if t in tf
            )
            scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scores[:k] if s > 0]


@lru_cache(maxsize=1)
def _index() -> BM25:
    return BM25(load_chunks())


def search_docs(query: str, k: int = 3) -> dict:
    hits = _index().search(query, k)
    if not hits:
        return {"results": [], "note": "no matching section"}
    return {"results": [{"id": c.id, "title": c.title, "score": round(s, 3), "text": c.text} for c, s in hits]}


SEARCH_DOCS = Tool(
    "search_docs",
    "Search the knowledge base (scenario design policy, model inventory, data dictionary, Federal Reserve 2026 "
    "scenario reference points, glossary, operating rules). Returns the best-matching sections with ids; cite "
    "them as [doc:<id>].",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    search_docs,
)
