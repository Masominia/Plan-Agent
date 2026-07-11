from planagent import evals
from planagent.llm import ScriptedBackend


def test_eval_suite_scripted_backend():
    tasks, runs = evals.run_tasks(ScriptedBackend())
    assert tasks["answer correct"].all() and tasks["guardrail pass"].all() and tasks["tools as expected"].all()
    gr = evals.red_team_guardrail(runs)
    assert gr.loc["clean answers flagged (false positives)", "detected (default gate)"] == 0
    assert gr.loc["invented sentence", "detected (default gate)"] == 1
    assert (gr.loc[[i for i in gr.index if i.startswith("x")], "detected (default gate)"] > 0.8).all()


def test_sql_red_team_and_retrieval():
    sq = evals.red_team_sql()
    assert (sq["rate"] == 1).all()
    rt, _ = evals.retrieval_eval()
    assert rt.loc["recall@3", "value"] >= 0.9
