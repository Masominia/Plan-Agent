"""Tool registry: every number the agent may report comes out of one of these.

Each tool has a JSON-schema input (validated before execution), a pure Python
implementation, and returns a JSON-serialisable dict. Errors are returned as
data (``{"error": ...}``) so the model can correct itself instead of crashing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import jsonschema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., dict]

    def spec(self) -> dict:
        """Provider-neutral spec (Anthropic-style); backends translate it."""
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


@dataclass
class ToolResult:
    name: str
    args: dict
    output: dict
    ok: bool
    seconds: float

    def to_json(self) -> str:
        return json.dumps(self.output, default=float, separators=(",", ":"))


class Registry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> ToolResult:
        t0 = time.perf_counter()
        if name not in self.tools:
            return ToolResult(name, args, {"error": f"unknown tool '{name}'"}, False, 0.0)
        tool = self.tools[name]
        try:
            jsonschema.validate(args, tool.schema)
        except jsonschema.ValidationError as e:
            return ToolResult(name, args, {"error": f"invalid arguments: {e.message}"}, False, time.perf_counter() - t0)
        try:
            out = tool.fn(**args)
            ok = "error" not in out
        except Exception as e:  # noqa: BLE001 - surfaced to the model as data
            out, ok = {"error": f"{type(e).__name__}: {e}"}, False
        return ToolResult(name, args, out, ok, time.perf_counter() - t0)


def default_registry() -> Registry:
    from .calc import CALCULATE
    from .docs import SEARCH_DOCS
    from .lineitem import PROJECT_LINE_ITEM
    from .scenario import RUN_SCENARIO
    from .sql import DESCRIBE_SERIES, QUERY_DATA

    return Registry([QUERY_DATA, DESCRIBE_SERIES, CALCULATE, RUN_SCENARIO, PROJECT_LINE_ITEM, SEARCH_DOCS])
