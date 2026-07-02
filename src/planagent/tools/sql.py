"""Read-only SQL access with defence in depth.

1. The database is opened read-only (``mode=ro`` URI).
2. A SQLite authorizer allows only SELECT, reads of whitelisted tables and a
   whitelist of functions; every write, ATTACH, PRAGMA or DDL is denied by the
   engine itself, whatever the SQL text looks like.
3. One statement per call (enforced by sqlite3), row cap, and a time limit
   through the progress handler.
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd

from ..data import DB, ensure_db
from . import Tool

ALLOWED_TABLES = {"macro_quarterly", "sp500_fundamentals", "recessions", "series_catalog"}
ALLOWED_FUNCTIONS = {
    "min",
    "max",
    "avg",
    "sum",
    "count",
    "abs",
    "round",
    "substr",
    "lower",
    "upper",
    "length",
    "coalesce",
    "ifnull",
    "exp",
    "ln",
    "log",
    "sqrt",
    "power",
    "lag",
    "lead",
    "row_number",
    "rank",
    "first_value",
    "last_value",
    "instr",
    "cast",
    "printf",
    "total",
}
MAX_ROWS = 500
TIME_LIMIT_S = 2.0


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_RECURSIVE:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _connect() -> sqlite3.Connection:
    ensure_db(DB)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.set_authorizer(_authorizer)
    start = time.perf_counter()
    con.set_progress_handler(lambda: int(time.perf_counter() - start > TIME_LIMIT_S), 10_000)
    return con


def query_data(sql: str, max_rows: int = 200) -> dict:
    max_rows = min(int(max_rows), MAX_ROWS)
    con = _connect()
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    except (sqlite3.DatabaseError, sqlite3.Warning) as e:
        return {"error": f"query rejected: {e}"}
    finally:
        con.close()
    truncated = len(rows) > max_rows
    rows = [[round(v, 6) if isinstance(v, float) else v for v in r] for r in rows[:max_rows]]
    return {"columns": cols, "rows": rows, "row_count": len(rows), "truncated": truncated}


def describe_series(table: str, column: str, start: str | None = None, end: str | None = None) -> dict:
    if table not in {"macro_quarterly", "sp500_fundamentals"}:
        return {"error": f"table must be macro_quarterly or sp500_fundamentals, got {table}"}
    con = _connect()
    try:
        cols = {d[0] for d in con.execute(f"SELECT * FROM {table} LIMIT 0").description}
        if column not in cols or column == "quarter":
            return {"error": f"unknown column {column}; available: {sorted(cols - {'quarter'})}"}
        df = pd.read_sql(f"SELECT quarter, {column} AS v FROM {table} ORDER BY quarter", con)
    finally:
        con.close()
    if start:
        df = df[df["quarter"] >= start]
    if end:
        df = df[df["quarter"] <= end]
    df = df.dropna()
    if df.empty:
        return {"error": "no data in range"}
    v = df["v"].to_numpy()
    i_min, i_max = int(np.argmin(v)), int(np.argmax(v))
    return {
        "table": table,
        "column": column,
        "first": df["quarter"].iloc[0],
        "last": df["quarter"].iloc[-1],
        "count": int(len(v)),
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=1)), 4),
        "min": round(float(v[i_min]), 4),
        "min_quarter": df["quarter"].iloc[i_min],
        "max": round(float(v[i_max]), 4),
        "max_quarter": df["quarter"].iloc[i_max],
        "latest": round(float(v[-1]), 4),
    }


QUERY_DATA = Tool(
    "query_data",
    "Run ONE read-only SQL SELECT against the planning database. Tables: macro_quarterly (quarter, gdp_growth, "
    "cpi_inflation, ust10y, sp500_return, vix_max, house_price_growth, wti_return, eurusd; 1990Q2-2024Q4), "
    "sp500_fundamentals (quarter, price, eps_ttm, dps_ttm, eps_growth, dps_growth; 1960Q1-2023Q2), recessions "
    "(peak, trough; NBER), series_catalog (table_name, column_name, description, unit, source). Quarters are text "
    "'YYYYQn' and sort correctly. Writes, PRAGMA and ATTACH are blocked.",
    {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "minLength": 6},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS},
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
    query_data,
)

DESCRIBE_SERIES = Tool(
    "describe_series",
    "Summary statistics of one column over an optional quarter range: count, mean, std, min and max with their "
    "quarters, latest value. Use it instead of writing aggregate SQL for simple questions.",
    {
        "type": "object",
        "properties": {
            "table": {"type": "string", "enum": ["macro_quarterly", "sp500_fundamentals"]},
            "column": {"type": "string"},
            "start": {"type": "string", "pattern": "^[12][0-9]{3}Q[1-4]$"},
            "end": {"type": "string", "pattern": "^[12][0-9]{3}Q[1-4]$"},
        },
        "required": ["table", "column"],
        "additionalProperties": False,
    },
    describe_series,
)
