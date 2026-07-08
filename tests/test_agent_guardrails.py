import json
from types import SimpleNamespace

import pytest

from planagent import guardrails
from planagent.agent import Agent
from planagent.llm import (
    AnthropicBackend,
    AssistantTurn,
    Backend,
    BedrockBackend,
    ScriptedBackend,
    from_converse,
    parse_scenario,
    to_converse,
)

# Guardrails
TOOLS = [
    {"earnings": {"median_change_pct": -52.0, "level_now": 181.17}, "summary": {"p_value": 2.54e-10}},
    {"expression": "(6.86/84.92-1)*100", "value": -91.9218087612},
]


def test_provenance_accepts_rounding_sign_and_percent():
    a = "Median EPS falls 52.0% from 181.17; p-value 2.54e-10; earnings fell 91.92% in 2008-09 over 9 quarters."
    g = guardrails.check(a, "", TOOLS, set())
    assert g.passed and g.n_claims == 4


@pytest.mark.parametrize("bad", ["Median EPS falls 53.0%", "p-value 3.1e-10", "level 181.71", "Revenue rose 17.3%"])
def test_provenance_rejects_unsupported(bad):
    assert not guardrails.check(bad, "", TOOLS, set()).passed


def test_labels_are_not_quantities():
    claims = [c.text for c in guardrails.extract_claims("5th to 95th percentile, 10Y yield, 2008Q4, -3.2%")]
    assert claims == ["-3.2%"]


def test_citations_must_be_retrieved():
    g = guardrails.check("See [doc:glossary#capital-terms].", "", [], {"glossary#capital-terms"})
    assert g.passed and g.n_citations == 1
    assert not guardrails.check("See [doc:made#up].", "", [], set()).passed


def test_question_numbers_are_allowed():
    assert guardrails.check("With equities down 58%, ...", "equities fall 58%", [], set()).passed


# Agent
def test_scripted_agent_end_to_end(tmp_path):
    r = Agent(ScriptedBackend(), audit_dir=tmp_path).run(
        "How far did S&P 500 earnings fall from peak to trough between 2007 and 2009?"
    )
    assert r.guardrail.passed and "-91.92%" in r.answer
    events = [json.loads(line)["event"] for line in open(r.audit_path)]
    assert events[0] == "run_start" and events[-1] == "run_end" and "tool_call" in events and "guardrail" in events


class LyingBackend(Backend):
    """Makes up a number first, then complies after the guardrail objects."""

    name, model = "lying", "test"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, tools):
        self.calls += 1
        last = messages[-1]["content"][0]
        if self.calls == 1:
            return AssistantTurn(
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "calculate",
                        "input": {"expression": "100 * (6.86 / 84.92 - 1)"},
                    }
                ],
                "tool_use",
            )
        if last.get("type") == "text" and last["text"].startswith("GUARDRAIL"):
            return AssistantTurn([{"type": "text", "text": "Earnings fell 91.92%."}])
        return AssistantTurn([{"type": "text", "text": "Earnings fell 95.5%."}])


def test_guardrail_forces_revision():
    r = Agent(LyingBackend(), audit_dir=None).run("How much did earnings fall?")
    assert r.revised and r.guardrail.passed and r.answer == "Earnings fell 91.92%."


def test_parse_scenario():
    assert parse_scenario("if GDP falls 8% for 2 quarters and equities fall 58%") == ([-8.0, -8.0], -58.0)
    assert parse_scenario("GDP path of -3, -2.5, -1, 0.5")[0] == [-3.0, -2.5, -1.0, 0.5]
    assert parse_scenario("tell me a joke") == ([], None)


# Live backends (mocked)
class FakeAnthropic:
    def __init__(self):
        self.messages = self
        self.n = 0

    def create(self, **kw):
        self.n += 1
        assert kw["tools"] and kw["system"]
        if self.n == 1:
            blocks = [
                SimpleNamespace(
                    type="tool_use",
                    id="tu1",
                    name="describe_series",
                    input={"table": "sp500_fundamentals", "column": "eps_growth"},
                )
            ]
            return SimpleNamespace(
                content=blocks, stop_reason="tool_use", usage=SimpleNamespace(input_tokens=10, output_tokens=5)
            )
        res = json.loads(kw["messages"][-1]["content"][0]["content"])
        text = f"The worst quarter was {res['min_quarter']} at {res['min']}%."
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=20, output_tokens=8),
        )


def test_anthropic_backend_with_agent():
    r = Agent(AnthropicBackend("test-model", client=FakeAnthropic()), audit_dir=None).run("Worst EPS quarter?")
    assert r.guardrail.passed and "2008Q4" in r.answer and r.model_turns == 2


def test_bedrock_message_translation_round_trip():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": "calculate", "input": {"expression": "1+1"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": '{"value": 2.0}'}]},
    ]
    conv = to_converse(msgs)
    assert conv[1]["content"][0]["toolUse"]["name"] == "calculate"
    assert conv[2]["content"][0]["toolResult"]["content"][0]["json"] == {"value": 2.0}
    back = from_converse({"content": [{"text": "ok"}, {"toolUse": {"toolUseId": "y", "name": "n", "input": {}}}]})
    assert back[0] == {"type": "text", "text": "ok"} and back[1]["id"] == "y"


class FakeBedrock:
    def converse(self, **kw):
        assert kw["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
        return {"output": {"message": {"content": [{"text": "No data needed."}]}}, "stopReason": "end_turn"}


def test_bedrock_backend_with_agent():
    r = Agent(BedrockBackend("model-id", client=FakeBedrock()), audit_dir=None).run("hello")
    assert r.answer == "No data needed."
