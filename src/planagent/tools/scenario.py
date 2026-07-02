"""Scenario expansion: a narrative (GDP path, equity move) -> full macro paths.

VAR(1) on (GDP growth, CPI inflation, 10Y yield, S&P 500 return, VIX max),
1990Q2-2024Q4, with 2020Q2-Q3 excluded from estimation (as dependent variable
or lag). Narratives are linear restrictions on the stacked future path; the
conditional forecast is the Gaussian conditional mean (Waggoner-Zha), and the
Mahalanobis distance of the implied shocks gives a chi-square plausibility.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

from ..data import macro_frame
from . import Tool

VARS = ["gdp_growth", "cpi_inflation", "ust10y", "sp500_return", "vix_max"]
EXCLUDE = {"2020Q2", "2020Q3"}


@lru_cache(maxsize=1)
def _model():
    df = macro_frame()[VARS].dropna()
    Y = df.to_numpy()
    q = df.index.astype(str)
    X, Z = [], []
    for t in range(1, len(Y)):
        if q[t] in EXCLUDE or q[t - 1] in EXCLUDE:
            continue
        X.append(np.r_[1.0, Y[t - 1]])
        Z.append(Y[t])
    X, Z = np.array(X), np.array(Z)
    B, *_ = np.linalg.lstsq(X, Z, rcond=None)
    U = Z - X @ B
    sigma = U.T @ U / (len(Z) - X.shape[1])
    return B[0], B[1:].T, sigma, Y[-1], str(df.index[-1]), len(Z)


def _scenario(gdp_path: list[float], equity_change_pct: float | None, horizon: int):
    c, A, S, y0, last, nobs = _model()
    k, H = len(VARS), horizon
    F, y = [], y0.copy()
    for _ in range(H):
        y = c + A @ y
        F.append(y)
    F = np.concatenate(F)
    Psi = [np.eye(k)]
    for _ in range(1, H):
        Psi.append(A @ Psi[-1])
    M = np.zeros((H * k, H * k))
    for h in range(H):
        for j in range(h + 1):
            M[h * k : (h + 1) * k, j * k : (j + 1) * k] = Psi[h - j]
    rows, vals = [], []
    for h, g in enumerate(gdp_path[:H]):
        r = np.zeros(H * k)
        r[h * k + 0] = 1.0
        rows.append(r)
        vals.append(g)
    if equity_change_pct is not None:  # cumulative log change over the first three quarters
        r = np.zeros(H * k)
        for h in range(min(3, H)):
            r[h * k + 3] = 1.0
        rows.append(r)
        vals.append(100 * np.log1p(equity_change_pct / 100))
    Omega = np.kron(np.eye(H), S)
    if rows:
        Ry, rv = np.array(rows), np.array(vals)
        R = Ry @ M
        G = R @ Omega @ R.T
        gap = rv - Ry @ F
        sol = np.linalg.solve(G, gap)
        u = Omega @ R.T @ sol
        d2, q = float(gap @ sol), len(rows)
    else:
        u, d2, q = np.zeros(H * k), 0.0, 0
    paths = (F + M @ u).reshape(H, k)
    return paths, d2, q, last, nobs


def run_scenario(gdp_path: list[float], equity_change_pct: float | None = None, horizon: int = 9) -> dict:
    paths, d2, q, last, nobs = _scenario(gdp_path, equity_change_pct, horizon)
    eq_idx = 100 * np.exp(np.cumsum(paths[:, 3]) / 100)
    lvl = np.r_[100.0, eq_idx]
    p = float(stats.chi2.sf(d2, q)) if q else 1.0
    gdp_lvl = np.exp(np.r_[0.0, np.cumsum(paths[:, 0] / 400)])
    return {
        "jump_off": last,
        "estimation_quarters": nobs,
        "quarters_ahead": list(range(1, horizon + 1)),
        "gdp_growth": np.round(paths[:, 0], 2).tolist(),
        "cpi_inflation": np.round(paths[:, 1], 2).tolist(),
        "ust10y": np.round(paths[:, 2], 2).tolist(),
        "sp500_return": np.round(paths[:, 3], 2).tolist(),
        "vix_max": np.round(paths[:, 4], 1).tolist(),
        "summary": {
            "gdp_peak_to_trough_pct": round(float((gdp_lvl / np.maximum.accumulate(gdp_lvl)).min() - 1) * 100, 2),
            "equity_peak_to_trough_pct": round(float((lvl / np.maximum.accumulate(lvl)).min() - 1) * 100, 2),
            "vix_peak": round(float(paths[:, 4].max()), 1),
            "ust10y_min": round(float(paths[:, 2].min()), 2),
            "ust10y_max": round(float(paths[:, 2].max()), 2),
            "inflation_peak": round(float(paths[:, 1].max()), 2),
            "chi2_distance": round(d2, 2),
            "restrictions": q,
            "p_value": float(f"{p:.3g}"),
        },
        "units": "GDP and CPI % annualised; 10Y %; S&P return % per quarter (log); VIX quarterly max",
    }


SCHEMA = {
    "type": "object",
    "properties": {
        "gdp_path": {
            "type": "array",
            "items": {"type": "number", "minimum": -40, "maximum": 40},
            "minItems": 1,
            "maxItems": 9,
            "description": "Real GDP growth, % annualised, for quarters 1, 2, ... (rest free)",
        },
        "equity_change_pct": {
            "type": "number",
            "minimum": -95,
            "maximum": 200,
            "description": "Cumulative S&P 500 change over the first three quarters, %",
        },
        "horizon": {"type": "integer", "minimum": 1, "maximum": 9},
    },
    "required": ["gdp_path"],
    "additionalProperties": False,
}

RUN_SCENARIO = Tool(
    "run_scenario",
    "Expand a macro narrative into consistent 9-quarter paths with a VAR(1) estimated on 1990-2024 data. Inputs: "
    "GDP growth path (% annualised) and optionally the cumulative equity change over the first three quarters. "
    "Returns paths for GDP, inflation, 10Y yield, equity returns and VIX, plus a chi-square plausibility p-value "
    "(how likely such a narrative is under the model).",
    SCHEMA,
    run_scenario,
)
