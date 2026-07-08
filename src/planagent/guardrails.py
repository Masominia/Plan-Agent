"""Guardrails applied to every draft answer before it is released.

* **Numeric provenance**: every number in the answer must be traceable to a
  tool output or to the user's question, allowing for rounding to the shown
  precision, sign flips ("fell 26%" for -26) and percent/fraction scaling.
  Years, quarter labels and small counts are exempt. Each match is graded:
  *strong* when the words around the number in the answer share a term with
  the source's context (JSON key path, SQL column, neighbouring words), *weak*
  when only the value matches. The default gate requires every number to be at
  least weakly supported; the strict gate requires strong support.
* **Citations**: every [doc:<id>] must refer to a section actually returned by
  search_docs in this run.

A violation does not silently pass: the agent is asked to revise once, and an
answer that still fails is released only with the violations attached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# a quantity: not glued to letters on either side, so labels such as "95th", "10Y" or "Q2" are not quantities
NUM_RE = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\w])", re.IGNORECASE)
QUARTER_RE = re.compile(r"\b(19|20)\d{2}\s?Q[1-4]\b")
CITE_RE = re.compile(r"\[doc:([a-z0-9_\-]+#[a-z0-9\-]+)\]")


SYNONYMS = {
    "eps": "earnin",
    "earnin": "earnin",
    "dps": "divide",
    "divide": "divide",
    "sp500": "equity",
    "equity": "equity",
    "equiti": "equity",
    "stock": "equity",
    "s": "",
    "ust10y": "yield",
    "10y": "yield",
    "yield": "yield",
    "vix": "vix",
    "gdp": "gdp",
    "cpi": "inflat",
    "inflat": "inflat",
    "p": "pvalue",
    "pvalue": "pvalue",
    "chi2": "chi",
    "chi": "chi",
    "distan": "chi",
    "drawdo": "drawdo",
    "trough": "drawdo",
    "peak": "peak",
    "level": "level",
    "change": "change",
    "chg": "change",
    "fell": "change",
    "fall": "change",
    "declin": "change",
    "path": "path",
    "percen": "percen",
    "p5": "percen",
    "p95": "percen",
    "house": "house",
    "unempl": "unempl",
    "cet1": "cet1",
    "capita": "cet1",
}


DERIVED = frozenset({"change", "ratio", "differ", "increa", "growth", "rise", "percen", "drawdo", "multip"})


def _ctx(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return {SYNONYMS.get(t[:6], t[:6]) for t in toks if len(t) > 1 and not t.isdigit()} - {""}


def _numbers_in(obj, out: list, ctx: frozenset = frozenset()) -> list:
    """Collect (value, context tokens): JSON key paths for structured outputs, SQL column
    names for query rows, neighbouring words for numbers inside text."""
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append((float(obj), ctx))
    elif isinstance(obj, str):
        for m in NUM_RE.finditer(obj):
            v = _parse(m.group())
            if v is not None:
                near = obj[max(0, m.start() - 60) : m.end() + 60]
                out.append((v, ctx | frozenset(_ctx(near))))
    elif isinstance(obj, dict):
        if "expression" in obj:  # calculate result: a derived number, described as a change / ratio / difference
            ctx = ctx | DERIVED
        if "columns" in obj and "rows" in obj:  # query_data result
            cols = [frozenset(_ctx(c)) for c in obj["columns"]]
            for row in obj["rows"]:
                rctx = frozenset().union(*[_ctx(str(x)) for x in row if isinstance(x, str)])
                for c, v in zip(cols, row, strict=False):
                    _numbers_in(v, out, ctx | c | rctx)
        for k, v in obj.items():
            if k not in ("columns", "rows"):
                _numbers_in(v, out, ctx | frozenset(_ctx(str(k))))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out, ctx)
    return out


def _parse(tok: str) -> float | None:
    t = tok.replace("−", "-").replace(",", "").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


@dataclass
class NumberClaim:
    text: str
    value: float
    decimals: int
    context: str
    tol: float = 0.0  # half a unit in the last shown digit


def extract_claims(answer: str) -> list[NumberClaim]:
    masked = QUARTER_RE.sub(lambda m: "Q" * len(m.group()), answer)  # quarter labels are not quantities
    masked = CITE_RE.sub(lambda m: "C" * len(m.group()), masked)
    claims = []
    for m in NUM_RE.finditer(masked):
        tok = m.group()
        v = _parse(tok)
        if v is None:
            continue
        core = tok.rstrip("%").replace(",", "").lower()
        mant, _, exp = core.partition("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        tol = 0.5 * 10 ** (-dec) * 10 ** (int(exp) if exp else 0)
        s, e = max(0, m.start() - 40), min(len(answer), m.end() + 40)
        claims.append(NumberClaim(tok, v, dec, answer[s:e].replace("\n", " "), tol))
    return claims


def _exempt(c: NumberClaim) -> bool:
    if c.decimals == 0 and not c.text.endswith("%") and "e" not in c.text.lower() and c.value == round(c.value):
        if 1900 <= abs(c.value) <= 2100:  # calendar years
            return True
        if abs(c.value) <= 12:  # small counts: quarters, horizons, ordinals
            return True
    return False


def _value_match(c: NumberClaim, v: float) -> bool:
    tol = c.tol + 1e-12
    for cand in (v, 100 * v, v / 100):
        if abs(abs(c.value) - abs(cand)) <= tol:  # sign-insensitive: "fell 26%" for -26
            return True
        if abs(cand) >= 1000 and abs(abs(c.value) - abs(cand)) <= max(tol, 5e-4 * abs(cand)):
            return True
    return False


def support_level(c: NumberClaim, sources: list) -> str:
    """'strong': value and context agree; 'weak': value only; 'none'."""
    matches = [ctx for v, ctx in sources if _value_match(c, v)]
    if not matches:
        return "none"
    cctx = _ctx(c.context)
    return "strong" if any(cctx & ctx for ctx in matches) else "weak"


def supported(c: NumberClaim, sources: list) -> bool:
    return support_level(c, sources) != "none"


@dataclass
class GuardrailReport:
    unsupported: list[NumberClaim] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)
    n_claims: int = 0
    n_citations: int = 0
    weak: list[NumberClaim] = field(default_factory=list)  # value matches, but not in a matching context

    @property
    def passed(self) -> bool:
        return not self.unsupported and not self.bad_citations

    @property
    def passed_strict(self) -> bool:
        return self.passed and not self.weak

    def summary(self) -> str:
        if self.passed:
            w = f" ({len(self.weak)} matched by value only)" if self.weak else ""
            nums = f"{self.n_claims} number{'s' if self.n_claims != 1 else ''}"
            cits = f"{self.n_citations} citation{'s' if self.n_citations != 1 else ''}"
            return f"PASS: {nums} traced to tool outputs or the question{w}, {cits} valid"
        parts = []
        if self.unsupported:
            parts.append("unsupported numbers: " + "; ".join(f"{c.text} (…{c.context}…)" for c in self.unsupported))
        if self.bad_citations:
            parts.append("citations not retrieved in this run: " + ", ".join(self.bad_citations))
        return "FAIL: " + " | ".join(parts)


def check(answer: str, question: str, tool_outputs: list[dict], retrieved_ids: set[str]) -> GuardrailReport:
    sources = _numbers_in(tool_outputs, [])
    sources += _numbers_in(question, [])
    claims = [c for c in extract_claims(answer) if not _exempt(c)]
    levels = {id(c): support_level(c, sources) for c in claims}
    bad = [c for c in claims if levels[id(c)] == "none"]
    weak = [c for c in claims if levels[id(c)] == "weak"]
    cites = CITE_RE.findall(answer)
    return GuardrailReport(bad, sorted({c for c in cites if c not in retrieved_ids}), len(claims), len(cites), weak)
