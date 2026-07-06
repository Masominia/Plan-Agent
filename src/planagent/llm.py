"""Model backends behind one interface.

Conversation format (provider-neutral, Anthropic-style content blocks):
``{"role": "user" | "assistant", "content": [block, ...]}`` with blocks
``{"type": "text", "text"}``, ``{"type": "tool_use", "id", "name", "input"}``,
``{"type": "tool_result", "tool_use_id", "content", "is_error"}``.

* ``AnthropicBackend``: Claude via the Anthropic Messages API (tool use).
* ``BedrockBackend``: any tool-capable model on AWS Bedrock via the Converse API.
* ``ScriptedBackend``: a deterministic, rule-based planner with the same
  interface. It is *not* a language model: it exists so the whole pipeline
  (tools, guardrails, audit, evaluation) runs offline and in CI, and so the
  answers it produces can serve as a regression baseline for live models.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class AssistantTurn:
    blocks: list[dict]
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.blocks if b["type"] == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.blocks if b["type"] == "text").strip()


class Backend:
    name = "base"
    model = ""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:  # pragma: no cover
        raise NotImplementedError


# Anthropic
class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 2000, client=None):
        self.model = model or os.environ.get("PLANAGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        self.client = client

    def complete(self, system, messages, tools):
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools
        )
        blocks = []
        for b in r.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        usage = {
            "input_tokens": getattr(r.usage, "input_tokens", None),
            "output_tokens": getattr(r.usage, "output_tokens", None),
        }
        return AssistantTurn(blocks, r.stop_reason, usage)


# Bedrock
def to_converse(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = []
        for b in m["content"]:
            if b["type"] == "text":
                content.append({"text": b["text"]})
            elif b["type"] == "tool_use":
                content.append({"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": b["input"]}})
            elif b["type"] == "tool_result":
                try:
                    body = [{"json": json.loads(b["content"])}]
                except (TypeError, ValueError):
                    body = [{"text": str(b["content"])}]
                content.append(
                    {
                        "toolResult": {
                            "toolUseId": b["tool_use_id"],
                            "content": body,
                            "status": "error" if b.get("is_error") else "success",
                        }
                    }
                )
        out.append({"role": m["role"], "content": content})
    return out


def from_converse(message: dict) -> list[dict]:
    blocks = []
    for c in message.get("content", []):
        if "text" in c:
            blocks.append({"type": "text", "text": c["text"]})
        elif "toolUse" in c:
            t = c["toolUse"]
            blocks.append({"type": "tool_use", "id": t["toolUseId"], "name": t["name"], "input": t["input"]})
    return blocks


class BedrockBackend(Backend):
    name = "bedrock"

    def __init__(self, model: str | None = None, region: str | None = None, client=None, max_tokens: int = 2000):
        self.model = model or os.environ.get("PLANAGENT_BEDROCK_MODEL", "")
        if not self.model:
            raise ValueError("set a Bedrock model id with --model or PLANAGENT_BEDROCK_MODEL")
        self.max_tokens = max_tokens
        if client is None:
            import boto3  # optional dependency

            client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION"))
        self.client = client

    def complete(self, system, messages, tools):
        cfg = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t["name"],
                        "description": t["description"],
                        "inputSchema": {"json": t["input_schema"]},
                    }
                }
                for t in tools
            ]
        }
        r = self.client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=to_converse(messages),
            toolConfig=cfg,
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        stop = r.get("stopReason", "end_turn")
        return AssistantTurn(
            from_converse(r["output"]["message"]), "tool_use" if stop == "tool_use" else stop, r.get("usage", {})
        )


# Scripted
_NUM = r"[-+−]?\d+(?:\.\d+)?"


def parse_scenario(q: str) -> tuple[list[float], float | None]:
    """Extract a GDP path and an equity move from plain English (scripted planner only)."""
    ql = q.lower().replace("−", "-")
    gdp: list[float] = []
    m = re.search(r"gdp[^.;]*?\b(?:path|growth)\s*(?:of|=|:)?\s*((?:" + _NUM + r"\s*,\s*)+" + _NUM + ")", ql)
    if m:
        gdp = [float(x) for x in re.findall(_NUM, m.group(1))]
    else:
        m = re.search(r"gdp[^.;]*?(fall|drop|contract|decline|shrink)s?\w*\s+(?:by\s+)?(" + _NUM + r")\s*%", ql)
        if m:
            q_m = re.search(r"for\s+(\d+)\s+quarters", ql[m.end() :])
            n = int(q_m.group(1)) if q_m else 2
            gdp = [-abs(float(m.group(2)))] * n
        else:
            m = re.search(r"gdp[^.;]*?(grow|rise|expand)s?\w*\s+(?:by\s+|at\s+)?(" + _NUM + r")\s*%", ql)
            if m:
                gdp = [abs(float(m.group(2)))] * 4
    eq = None
    m = re.search(
        r"(equit\w*|stocks?|stock market|s&p\s*500)\s+(?:prices?\s+)?(fall|drop|decline|lose|crash)\w*\s+"
        r"(?:by\s+)?(" + _NUM + r")\s*%",
        ql,
    )
    if m:
        eq = -abs(float(m.group(3)))
    return gdp, eq


def _tool_results(messages) -> dict[str, list[dict]]:
    """name -> list of decoded outputs, from all tool_result blocks so far."""
    names = {}
    for m in messages:
        for b in m["content"]:
            if b["type"] == "tool_use":
                names[b["id"]] = b["name"]
    out: dict[str, list[dict]] = {}
    for m in messages:
        for b in m["content"]:
            if b["type"] == "tool_result":
                out.setdefault(names.get(b["tool_use_id"], "?"), []).append(json.loads(b["content"]))
    return out


def _fmt(x, d=1):
    return f"{x:.{d}f}"


class ScriptedBackend(Backend):
    """Deterministic planner with three intents (scenario, history, documentation)."""

    name = "scripted"
    model = "rule-based-planner-v1"

    def __init__(self):
        self._n = 0

    def _use(self, name, args):
        self._n += 1
        return {"type": "tool_use", "id": f"call_{self._n}", "name": name, "input": args}

    def complete(self, system, messages, tools):
        question = messages[0]["content"][0]["text"]
        ql = question.lower()
        res = _tool_results(messages)
        last_user = messages[-1]["content"][0]
        if last_user["type"] == "text" and last_user["text"].startswith("GUARDRAIL"):
            return AssistantTurn(
                [
                    {
                        "type": "text",
                        "text": "I could not support every number with a tool output, "
                        "so I am not releasing a quantitative answer.",
                    }
                ]
            )
        gdp, eq = parse_scenario(question)
        if gdp:
            return self._scenario(question, gdp, eq, res)
        if any(
            k in ql
            for k in (
                "peak",
                "trough",
                "worst",
                "largest",
                "decline",
                "fell",
                "history",
                "since",
                "between",
                "highest",
                "lowest",
                "average",
            )
        ):
            return self._history(question, ql, res)
        if any(k in ql for k in ("how", "why", "what", "explain", "policy", "define", "which")):
            return self._docs(question, res)
        return AssistantTurn(
            [
                {
                    "type": "text",
                    "text": "This question is outside what the planning tools cover, "
                    "so I cannot answer it from the available data.",
                }
            ]
        )

    # -- intents
    def _scenario(self, q, gdp, eq, res):
        if "project_line_item" not in res:
            args = {"gdp_path": gdp}
            if eq is not None:
                args["equity_change_pct"] = eq
            return AssistantTurn(
                [
                    self._use("project_line_item", args),
                    self._use("search_docs", {"query": "plausibility severity scenario p-value"}),
                ],
                "tool_use",
            )
        r = res["project_line_item"][0]
        s, e, d = r["scenario_summary"], r["earnings"], r["dividends"]
        doc = res["search_docs"][0]["results"][0]["id"] if res.get("search_docs") else None
        p = s["p_value"]
        band = "baseline-like" if p > 0.1 else ("adverse" if p > 1e-3 else "severe, beyond historical norms")
        lines = [
            f"**Scenario result** (jump-off {r['jump_off']}, {r['horizon_quarters']} quarters, {r['paths']} paths).",
            "",
            f"* Macro expansion: GDP peak-to-trough {_fmt(s['gdp_peak_to_trough_pct'], 2)}%, equities "
            f"{_fmt(s['equity_peak_to_trough_pct'], 2)}%, VIX peak {_fmt(s['vix_peak'])}, 10Y yield between "
            f"{_fmt(s['ust10y_min'], 2)}% and {_fmt(s['ust10y_max'], 2)}%.",
            f"* S&P 500 earnings: from {_fmt(e['level_now'], 2)} USD, median change {_fmt(e['median_change_pct'], 2)}% "
            f"(5th to 95th percentile {_fmt(e['p5_change_pct'], 2)}% to {_fmt(e['p95_change_pct'], 2)}%), median "
            f"drawdown {_fmt(e['median_drawdown_pct'], 2)}%.",
            f"* Dividends: from {_fmt(d['level_now'], 2)} USD, median change {_fmt(d['median_change_pct'], 2)}%, "
            f"5% worst drawdown {_fmt(d['p5_drawdown_pct'], 2)}%: dividends follow earnings with a lag and smooth.",
            f"* Plausibility: chi-square distance {_fmt(s['chi2_distance'], 2)} on {s['restrictions']} restrictions, "
            f"p-value {p:.3g} ({band})" + (f" [doc:{doc}]." if doc else "."),
        ]
        return AssistantTurn([{"type": "text", "text": "\n".join(lines)}])

    def _history(self, q, ql, res):
        item, col = ("dividends", "dps_ttm") if "dividend" in ql else ("earnings", "eps_ttm")
        yrs = re.findall(r"\b(19[6-9]\d|20[0-2]\d)\b", ql)
        start, end = (f"{min(yrs)}Q1", f"{max(yrs)}Q4") if len(yrs) >= 2 else ("2007Q1", "2009Q4")
        if "query_data" not in res:
            sql = (
                f"SELECT quarter, {col} FROM sp500_fundamentals WHERE quarter BETWEEN '{start}' AND '{end}' "
                f"ORDER BY quarter"
            )
            return AssistantTurn([self._use("query_data", {"sql": sql})], "tool_use")
        rows = res["query_data"][0]["rows"]
        if not rows:
            return AssistantTurn([{"type": "text", "text": f"No {item} data between {start} and {end}."}])
        vals = [r[1] for r in rows]
        i_peak = max(range(len(vals)), key=lambda i: vals[i])
        i_tr = min(range(i_peak, len(vals)), key=lambda i: vals[i])
        if "calculate" not in res:
            expr = f"({vals[i_tr]} / {vals[i_peak]} - 1) * 100"
            return AssistantTurn([self._use("calculate", {"expression": expr})], "tool_use")
        chg = res["calculate"][0]["value"]
        txt = (
            f"S&P 500 trailing-12-month {item} per share peaked at {vals[i_peak]} USD in {rows[i_peak][0]} and "
            f"fell to {vals[i_tr]} USD in {rows[i_tr][0]}, a change of {_fmt(chg, 2)}% (table sp500_fundamentals, "
            f"{start} to {end})."
        )
        return AssistantTurn([{"type": "text", "text": txt}])

    def _docs(self, q, res):
        if "search_docs" not in res:
            return AssistantTurn([self._use("search_docs", {"query": q, "k": 3})], "tool_use")
        hits = res["search_docs"][0]["results"]
        if not hits:
            return AssistantTurn([{"type": "text", "text": "The knowledge base has no section on this."}])
        top = hits[0]
        sentences = re.split(r"(?<=[.!?])\s+", top["text"])
        return AssistantTurn([{"type": "text", "text": " ".join(sentences[:3]) + f" [doc:{top['id']}]"}])


def make_backend(name: str, model: str | None = None) -> Backend:
    if name == "scripted":
        return ScriptedBackend()
    if name == "anthropic":
        return AnthropicBackend(model)
    if name == "bedrock":
        return BedrockBackend(model)
    raise ValueError(f"unknown backend {name}")
