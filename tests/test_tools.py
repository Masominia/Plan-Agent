import numpy as np
import pytest

from planagent.tools import default_registry
from planagent.tools.calc import calculate
from planagent.tools.docs import load_chunks, search_docs, tokenize
from planagent.tools.lineitem import project_line_item
from planagent.tools.scenario import run_scenario
from planagent.tools.sql import describe_series, query_data


def test_registry_validates_arguments():
    reg = default_registry()
    assert set(reg.tools) == {
        "query_data",
        "describe_series",
        "calculate",
        "run_scenario",
        "project_line_item",
        "search_docs",
    }
    bad = reg.call("run_scenario", {"gdp_path": "minus eight"})
    assert not bad.ok and "invalid arguments" in bad.output["error"]
    assert not reg.call("nope", {}).ok
    good = reg.call("calculate", {"expression": "1 + 2"})
    assert good.ok and good.output["value"] == 3.0


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE recessions",
        "PRAGMA table_info(recessions)",
        "SELECT * FROM sqlite_master",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "SELECT 1; SELECT 2",
        "SELECT load_extension('x')",
        "UPDATE sp500_fundamentals SET eps_ttm = 0",
    ],
)
def test_sql_guard_blocks(sql):
    assert "error" in query_data(sql)


def test_sql_returns_data_and_caps_rows():
    r = query_data("SELECT quarter, eps_ttm FROM sp500_fundamentals ORDER BY quarter", max_rows=5)
    assert r["row_count"] == 5 and r["truncated"] and r["columns"] == ["quarter", "eps_ttm"]
    r = query_data("SELECT eps_ttm FROM sp500_fundamentals WHERE quarter = '2009Q1'")
    assert r["rows"][0][0] == pytest.approx(6.86)


def test_describe_series():
    d = describe_series("sp500_fundamentals", "eps_growth")
    assert d["min_quarter"] == "2008Q4" and d["count"] == 254
    assert "error" in describe_series("sp500_fundamentals", "nope")


def test_calculator_is_safe():
    assert calculate("(6.86 / 84.92 - 1) * 100")["value"] == pytest.approx(-91.9218, abs=1e-4)
    for expr in ("__import__('os')", "open('x')", "2 ** 1000", "[x for x in range(3)]", "a + 1"):
        assert "error" in calculate(expr)


def test_scenario_honours_anchors_and_scores_plausibility():
    r = run_scenario([-8, -7, -4, -1], -58)
    assert r["gdp_growth"][:4] == [-8.0, -7.0, -4.0, -1.0]
    assert np.exp(sum(r["sp500_return"][:3]) / 100) - 1 == pytest.approx(-0.58, abs=0.002)
    assert r["summary"]["p_value"] < 1e-6 and r["summary"]["vix_peak"] > 45
    base = run_scenario([2.0])
    assert base["summary"]["p_value"] > 0.1


def test_line_items_respond_to_scenarios():
    bad = project_line_item([-8, -7, -4, -1], -58, paths=2000)
    good = project_line_item([3, 3, 3, 3], paths=2000)
    assert bad["earnings"]["median_change_pct"] < good["earnings"]["median_change_pct"]
    # dividends are stickier than earnings
    assert abs(bad["dividends"]["median_change_pct"]) < abs(bad["earnings"]["median_change_pct"])


def test_docs_index_and_search():
    chunks = load_chunks()
    assert len({c.id for c in chunks}) == len(chunks) >= 20
    hits = search_docs("stress capital buffer definition")["results"]
    assert hits[0]["id"] == "glossary#capital-terms"
    assert tokenize("Plausibility plausible") == ["plausi", "plausi"]
