"""The agent: a small, explicit state graph with an audit trail.

    plan/act --(tool calls)--> execute tools --> plan/act ...
         \\--(draft answer)--> verify (guardrails) --pass--> release
                                          \\--fail--> revise once --> verify --> release with flags

Kept framework-free on purpose (each node is a few lines and fully testable);
the same nodes map one-to-one onto a LangGraph StateGraph or a Bedrock
AgentCore runtime.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import guardrails
from .data import DB, ROOT, ensure_db
from .llm import Backend
from .tools import Registry, default_registry

SYSTEM_PROMPT = """You are a planning and scenario-analysis assistant for a bank's Planning & Analysis team.
Rules:
1. Every number you state must come from a tool result or from the user's question. Use the calculate tool for any
   derived number (differences, ratios, percentage changes); never do arithmetic yourself.
2. For data questions use query_data (one SELECT) or describe_series. For scenarios use run_scenario or
   project_line_item. For policy, methodology or reference points use search_docs and cite sections as [doc:<id>].
3. State units and periods. Keep answers short: a summary sentence, then bullet points.
4. If the tools cannot answer the question, say so. Do not give investment advice.
Your answer is checked automatically: numbers that cannot be traced to tool outputs are rejected."""


class AuditLog:
    def __init__(self, directory: Path | None):
        self.path = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.jsonl"

    def write(self, event: str, **payload):
        if self.path is None:
            return
        rec = {"ts": time.time(), "event": event, **payload}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class RunResult:
    question: str
    answer: str
    guardrail: guardrails.GuardrailReport
    tool_calls: list = field(default_factory=list)
    model_turns: int = 0
    revised: bool = False
    seconds: float = 0.0
    audit_path: str | None = None

    @property
    def status(self) -> str:
        return "released" if self.guardrail.passed else "released with guardrail flags"


class Agent:
    def __init__(
        self,
        backend: Backend,
        registry: Registry | None = None,
        max_turns: int = 8,
        audit_dir: Path | None = ROOT / "runs",
    ):
        self.backend, self.registry, self.max_turns = backend, registry or default_registry(), max_turns
        self.audit_dir = audit_dir

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        audit = AuditLog(self.audit_dir)
        db_hash = hashlib.sha256(Path(ensure_db(DB)).read_bytes()).hexdigest()[:16]
        audit.write(
            "run_start",
            question=question,
            backend=self.backend.name,
            model=self.backend.model,
            db_fingerprint=db_hash,
            tools=list(self.registry.tools),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        calls, outputs, retrieved = [], [], set()
        answer, revised, turns, report = "", False, 0, None
        while turns < self.max_turns:
            turn = self.backend.complete(SYSTEM_PROMPT, messages, self.registry.specs())
            turns += 1
            messages.append({"role": "assistant", "content": turn.blocks})
            audit.write("model_turn", n=turns, text=turn.text, tool_uses=turn.tool_uses, usage=turn.usage)
            if turn.tool_uses:  # Act
                results = []
                for tu in turn.tool_uses:
                    r = self.registry.call(tu["name"], tu["input"])
                    calls.append(r)
                    outputs.append(r.output)
                    if tu["name"] == "search_docs":
                        retrieved |= {h["id"] for h in r.output.get("results", [])}
                    payload = r.to_json()
                    audit.write(
                        "tool_call",
                        name=r.name,
                        args=r.args,
                        ok=r.ok,
                        seconds=round(r.seconds, 4),
                        output_sha=_sha(payload),
                        output=r.output,
                    )
                    results.append(
                        {"type": "tool_result", "tool_use_id": tu["id"], "content": payload, "is_error": not r.ok}
                    )
                messages.append({"role": "user", "content": results})
                continue
            answer = turn.text  # Verify
            report = guardrails.check(answer, question, outputs, retrieved)
            audit.write("guardrail", passed=report.passed, summary=report.summary())
            if report.passed or revised:
                break
            revised = True  # Revise once
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "GUARDRAIL: "
                            + report.summary()
                            + "\nRevise the answer: use only numbers from tool "
                            "outputs (call tools if needed) and only cite sections returned by search_docs.",
                        }
                    ],
                }
            )
        if report is None:
            report = guardrails.check(answer, question, outputs, retrieved)
        res = RunResult(
            question,
            answer,
            report,
            calls,
            turns,
            revised,
            time.perf_counter() - t0,
            str(audit.path) if audit.path else None,
        )
        audit.write("run_end", status=res.status, answer=answer, seconds=round(res.seconds, 3))
        return res
