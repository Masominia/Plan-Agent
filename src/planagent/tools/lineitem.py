"""Line-item projections under a scenario: S&P 500 earnings and dividends.

Earnings: ARX(2) on GDP growth, inflation, 10Y change and equity return,
Huber M-estimation on the common sample of both tables (1990Q2-2023Q2).
Dividends: Lintner (1956) partial adjustment to a target payout ratio,
driven by lagged earnings growth and simulated on top of each earnings path.
Shocks: stationary block bootstrap of historical residuals, both items drawn
on the same historical quarters.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..data import fundamentals_frame, macro_frame
from . import Tool
from .scenario import SCHEMA as SCEN_SCHEMA
from .scenario import run_scenario

X_COLS = ["gdp_growth", "cpi_inflation", "d_ust10y", "sp500_return"]


@lru_cache(maxsize=1)
def _models():
    f = fundamentals_frame()
    m = macro_frame()
    df = f.join(m[["gdp_growth", "cpi_inflation", "ust10y", "sp500_return"]], how="inner")
    df["d_ust10y"] = df["ust10y"].diff()
    g = df["eps_growth"]
    X = pd.concat([g.shift(1).rename("l1"), g.shift(2).rename("l2"), df[X_COLS]], axis=1)
    d = pd.concat([g.rename("y"), X], axis=1).dropna()
    Xe = sm.add_constant(d.drop(columns="y"))
    re = sm.RLM(d["y"], Xe, M=sm.robust.norms.HuberT()).fit()
    res_e = d["y"] - Xe @ re.params
    gD, gE = df["dps_growth"], df["eps_growth"]
    L = pd.DataFrame(
        {
            "d1": gD.shift(1),
            "d2": gD.shift(2),
            "e1": gE.shift(1),
            "e2": gE.shift(2),
            "e3": gE.shift(3),
            "e4": gE.shift(4),
            "payout": np.log(df["dps_ttm"] / df["eps_ttm"]).shift(1),
        }
    )
    dl = pd.concat([gD.rename("y"), L], axis=1).dropna()
    Xl = sm.add_constant(dl.drop(columns="y"))
    rl = sm.RLM(dl["y"], Xl, M=sm.robust.norms.HuberT()).fit()
    res_d = dl["y"] - Xl @ rl.params
    common = res_e.index.intersection(res_d.index)
    return re.params, rl.params, res_e.loc[common].to_numpy(), res_d.loc[common].to_numpy(), df, str(df.index[-1])


def project_line_item(
    gdp_path: list[float], equity_change_pct: float | None = None, horizon: int = 9, paths: int = 5000, seed: int = 0
) -> dict:
    be, bl, rE, rD, df, last = _models()
    sc = run_scenario(gdp_path, equity_change_pct, horizon)
    x = np.column_stack(
        [sc["gdp_growth"], sc["cpi_inflation"], np.diff(np.r_[df["ust10y"].iloc[-1], sc["ust10y"]]), sc["sp500_return"]]
    )
    rng = np.random.default_rng(seed)
    n, H, m = paths, horizon, len(rE)
    idx = np.zeros((n, H), dtype=int)
    idx[:, 0] = rng.integers(0, m, n)
    for t in range(1, H):
        new = rng.random(n) < 0.25
        idx[:, t] = np.where(new, rng.integers(0, m, n), (idx[:, t - 1] + 1) % m)
    gE = np.zeros((n, H))
    l1, l2 = np.full(n, df["eps_growth"].iloc[-1]), np.full(n, df["eps_growth"].iloc[-2])
    for t in range(H):
        y = be["const"] + be["l1"] * l1 + be["l2"] * l2 + x[t] @ be[X_COLS].to_numpy() + rE[idx[:, t]]
        gE[:, t] = y
        l1, l2 = y, l1
    hist_e = np.tile(df["eps_growth"].iloc[-4:].to_numpy(), (n, 1))
    allE = np.column_stack([hist_e, gE])
    d1, d2 = np.full(n, df["dps_growth"].iloc[-1]), np.full(n, df["dps_growth"].iloc[-2])
    lnD, lnE = np.full(n, np.log(df["dps_ttm"].iloc[-1])), np.full(n, np.log(df["eps_ttm"].iloc[-1]))
    gD = np.zeros((n, H))
    for t in range(H):
        y = (
            bl["const"]
            + bl["d1"] * d1
            + bl["d2"] * d2
            + sum(bl[f"e{j}"] * allE[:, 4 + t - j] for j in range(1, 5))
            + bl["payout"] * (lnD - lnE)
            + rD[idx[:, t]]
        )
        gD[:, t] = y
        d1, d2 = y, d1
        lnD, lnE = lnD + y / 100, lnE + gE[:, t] / 100

    def summ(g, level0):
        lvl = level0 * np.exp(np.cumsum(g, axis=1) / 100)
        full = np.column_stack([np.full(n, level0), lvl])
        dd = (full / np.maximum.accumulate(full, axis=1)).min(axis=1) - 1
        q = np.quantile(lvl[:, -1], [0.05, 0.5, 0.95])
        return {
            "level_now": round(float(level0), 2),
            "median_path": np.round(np.median(lvl, axis=0), 2).tolist(),
            "median_change_pct": round(float(q[1] / level0 - 1) * 100, 2),
            "p5_change_pct": round(float(q[0] / level0 - 1) * 100, 2),
            "p95_change_pct": round(float(q[2] / level0 - 1) * 100, 2),
            "median_drawdown_pct": round(float(np.median(dd)) * 100, 2),
            "p5_drawdown_pct": round(float(np.quantile(dd, 0.05)) * 100, 2),
        }

    return {
        "jump_off": last,
        "horizon_quarters": H,
        "paths": n,
        "earnings": summ(gE, float(df["eps_ttm"].iloc[-1])),
        "dividends": summ(gD, float(df["dps_ttm"].iloc[-1])),
        "scenario_summary": sc["summary"],
        "note": "S&P 500 trailing-12-month EPS and DPS per index share, USD; changes are over the horizon",
    }


PROJECT_LINE_ITEM = Tool(
    "project_line_item",
    "Project S&P 500 earnings (EPS) and dividends (DPS) over a macro scenario with Monte Carlo uncertainty. "
    "Same inputs as run_scenario (the scenario is expanded internally). Returns, for each item, the current "
    "level, median path, median / 5th / 95th percentile change over the horizon and drawdowns. Dividends are "
    "linked to the simulated earnings (Lintner payout model). Jump-off is the last fundamentals quarter.",
    {
        **SCEN_SCHEMA,
        "properties": {
            **SCEN_SCHEMA["properties"],
            "paths": {"type": "integer", "minimum": 500, "maximum": 50000},
            "seed": {"type": "integer"},
        },
    },
    project_line_item,
)
