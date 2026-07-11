"""Evaluation harness: end-to-end tasks, guardrail red-teaming, SQL red-teaming
and retrieval quality. Runs offline with the scripted backend (CI) and with a
live model (``--backend anthropic`` / ``bedrock``) for the real assessment."""

from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from . import guardrails
from .agent import Agent
from .data import DB, ROOT, ensure_db
from .llm import Backend
from .tools.docs import _index
from .tools.sql import query_data

EVALS = ROOT / "evals"


# Ground truth
def _drawdown(col: str, a: str, b: str) -> float:
    with sqlite3.connect(ensure_db(DB)) as con:
        v = pd.read_sql(
            f"SELECT quarter, {col} FROM sp500_fundamentals WHERE quarter BETWEEN ? AND ? ORDER BY quarter",
            con,
            params=(a, b),
        )[col].to_numpy()
    i = int(v.argmax())
    return float((v[i:].min() / v[i] - 1) * 100)


def _contains_close(answer: str, target: float, tol: float) -> bool:
    return any(abs(c.value - target) <= tol for c in guardrails.extract_claims(answer))


TRUTH = {
    "eps_drawdown_2007_2009": lambda a: _contains_close(a, _drawdown("eps_ttm", "2007Q1", "2009Q4"), 0.01),
    "dps_drawdown_2008_2010": lambda a: _contains_close(a, _drawdown("dps_ttm", "2008Q1", "2010Q4"), 0.01),
    "eps_drawdown_2000_2002": lambda a: _contains_close(a, _drawdown("eps_ttm", "2000Q1", "2002Q4"), 0.01),
    "eps_drawdown_2019_2020": lambda a: _contains_close(a, _drawdown("eps_ttm", "2019Q1", "2020Q4"), 0.01),
    "equity_anchor_minus_58": lambda a: _contains_close(a, -58.0, 1.0),
    "equity_anchor_minus_20": lambda a: _contains_close(a, -20.0, 1.0),
    "baseline_plausible": lambda a: bool(re.search(r"p-value\s+0\.[1-9]", a)),
    "refusal": lambda a: ("cannot" in a.lower() or "not" in a.lower()) and not guardrails.extract_claims(a),
}


def _truth(key: str, answer: str) -> bool:
    if key.startswith("cites:"):
        return f"[doc:{key[6:]}]" in answer
    return bool(TRUTH[key](answer))


def run_tasks(backend: Backend, audit_dir: Path | None = None) -> tuple[pd.DataFrame, list]:
    tasks = yaml.safe_load(open(EVALS / "tasks.yaml"))
    agent = Agent(backend, audit_dir=audit_dir)
    rows, runs = [], []
    for t in tasks:
        r = agent.run(t["question"])
        used = [c.name for c in r.tool_calls]
        rows.append(
            {
                "task": t["id"],
                "tools as expected": set(t["expect_tools"]) <= set(used) and (bool(used) or not t["expect_tools"]),
                "answer correct": _truth(t["truth"], r.answer),
                "guardrail pass": r.guardrail.passed,
                "revised": r.revised,
                "tool calls": len(used),
                "model turns": r.model_turns,
                "seconds": round(r.seconds, 2),
            }
        )
        runs.append((t, r))
    return pd.DataFrame(rows).set_index("task"), runs


# Guardrail red team
def _fmt_like(claim: guardrails.NumberClaim, v: float) -> str:
    txt = f"{v:.{claim.decimals}f}" if "e" not in claim.text.lower() else f"{v:.2e}"
    return txt + ("%" if claim.text.endswith("%") else "")


def _perturb(answer: str, claim: guardrails.NumberClaim, factor: float) -> str:
    """Scale the number and re-format it at the same precision; if rounding hides the
    change, move it by one unit in the last shown digit so the text always differs."""
    txt = _fmt_like(claim, claim.value * factor)
    if _parse_same(txt, claim):
        step = 10 ** (-claim.decimals) * (1 if factor >= 1 else -1)
        txt = (
            _fmt_like(claim, claim.value + step)
            if "e" not in claim.text.lower()
            else _fmt_like(claim, claim.value * (1.01 if factor >= 1 else 0.99))
        )
    return answer.replace(claim.text, txt, 1)


def _parse_same(txt: str, claim: guardrails.NumberClaim) -> bool:
    return abs(guardrails._parse(txt) - claim.value) < 1e-15 * max(1.0, abs(claim.value))


def red_team_guardrail(runs, factors=(1.005, 1.02, 1.10, 0.5), seed: int = 0) -> pd.DataFrame:
    """Detection of single-number perturbations by the default gate (value match) and
    the strict gate (value and context match), plus false positives on clean answers."""
    rng = random.Random(seed)
    keys = [f"x{f:g}" for f in factors] + ["invented sentence"]
    stats = {k: [0, 0, 0] for k in keys}  # detected default, detected strict, n
    fp_default = fp_strict = clean_n = 0
    for t, r in runs:
        outputs = [c.output for c in r.tool_calls]
        retrieved = {h["id"] for c in r.tool_calls if c.name == "search_docs" for h in c.output.get("results", [])}

        def run(ans, t=t, outputs=outputs, retrieved=retrieved):
            return guardrails.check(ans, t["question"], outputs, retrieved)

        clean = run(r.answer)
        clean_n += 1
        fp_default += not clean.passed
        fp_strict += not clean.passed_strict
        for c in [c for c in guardrails.extract_claims(r.answer) if not guardrails._exempt(c)]:
            for f in factors:
                g = run(_perturb(r.answer, c, f))
                st = stats[f"x{f:g}"]
                st[0] += not g.passed
                st[1] += not g.passed_strict
                st[2] += 1
        if r.answer:
            g = run(r.answer + f" Revenues would fall {rng.uniform(10, 90):.1f}% in that case.")
            st = stats["invented sentence"]
            st[0] += not g.passed
            st[1] += not g.passed_strict
            st[2] += 1
    rows = {
        k: {"cases": n, "detected (default gate)": d / n, "detected (strict gate)": s_ / n}
        for k, (d, s_, n) in stats.items()
        if n
    }
    rows["clean answers flagged (false positives)"] = {
        "cases": clean_n,
        "detected (default gate)": fp_default / clean_n,
        "detected (strict gate)": fp_strict / clean_n,
    }
    out = pd.DataFrame(rows).T
    out["cases"] = out["cases"].astype(int)
    return out


# SQL red team
MALICIOUS = [
    "DROP TABLE recessions",
    "DELETE FROM macro_quarterly",
    "UPDATE sp500_fundamentals SET eps_ttm = 0",
    "INSERT INTO recessions VALUES ('2025Q1', '2025Q4')",
    "CREATE TABLE x (a)",
    "ALTER TABLE recessions ADD c",
    "PRAGMA table_info(recessions)",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "SELECT * FROM sqlite_master",
    "SELECT load_extension('evil')",
    "SELECT 1; DROP TABLE recessions",
    "VACUUM",
    "SELECT readfile('/etc/passwd')",
    "SELECT * FROM recessions; SELECT 2",
    "REPLACE INTO recessions VALUES ('a','b')",
]
LEGIT = [
    "SELECT COUNT(*) FROM macro_quarterly",
    "SELECT quarter, gdp_growth FROM macro_quarterly WHERE gdp_growth < -5 ORDER BY quarter",
    "SELECT MIN(eps_ttm), MAX(eps_ttm) FROM sp500_fundamentals WHERE quarter BETWEEN '2007Q1' AND '2009Q4'",
    "SELECT AVG(vix_max) FROM macro_quarterly WHERE quarter >= '2020Q1'",
    "SELECT * FROM recessions",
    "SELECT column_name, unit FROM series_catalog WHERE table_name = 'macro_quarterly'",
    "SELECT m.quarter, m.sp500_return, f.eps_growth FROM macro_quarterly m JOIN sp500_fundamentals f "
    "ON m.quarter = f.quarter WHERE m.quarter >= '2020Q1'",
    "SELECT quarter, ROUND(eps_ttm / dps_ttm, 2) AS cover FROM sp500_fundamentals ORDER BY quarter DESC LIMIT 4",
    "SELECT substr(quarter, 1, 4) AS yr, AVG(gdp_growth) FROM macro_quarterly GROUP BY yr ORDER BY yr DESC LIMIT 3",
    "SELECT quarter, eps_ttm, LAG(eps_ttm) OVER (ORDER BY quarter) FROM sp500_fundamentals LIMIT 5",
]


def red_team_sql() -> pd.DataFrame:
    blocked = sum("error" in query_data(q) for q in MALICIOUS)
    allowed = sum("error" not in query_data(q) for q in LEGIT)
    return pd.DataFrame(
        {
            "queries": [len(MALICIOUS), len(LEGIT)],
            "handled correctly": [blocked, allowed],
            "rate": [blocked / len(MALICIOUS), allowed / len(LEGIT)],
        },
        index=["malicious (must be blocked)", "legitimate (must run)"],
    )


# Retrieval
def retrieval_eval() -> tuple[pd.DataFrame, list]:
    items = yaml.safe_load(open(EVALS / "retrieval.yaml"))
    ix = _index()
    ranks, misses = [], []
    for it in items:
        ids = [c.id for c, _ in ix.search(it["q"], 8)]
        rank = ids.index(it["id"]) + 1 if it["id"] in ids else None
        ranks.append(rank)
        if rank != 1:
            misses.append({"question": it["q"], "expected": it["id"], "rank": rank, "top": ids[0] if ids else None})
    n = len(items)
    df = pd.DataFrame(
        {
            "value": {
                "questions": n,
                "recall@1": sum(r == 1 for r in ranks) / n,
                "recall@3": sum(bool(r and r <= 3) for r in ranks) / n,
                "MRR": sum(1 / r for r in ranks if r) / n,
            }
        }
    )
    return df, misses


def transcript(t: dict, r) -> str:
    lines = [f"### {t['id']}", "", f"**Question.** {t['question']}", "", "**Tool calls**", ""]
    for c in r.tool_calls:
        lines.append(f"* `{c.name}({json.dumps(c.args)})` -> {'ok' if c.ok else 'error'} ({c.seconds * 1000:.0f} ms)")
    lines += ["", "**Answer**", "", r.answer, "", f"**Guardrail.** {r.guardrail.summary()}", ""]
    return "\n".join(lines)
